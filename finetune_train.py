import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time
import pdb
from config import PARAMS_CONFIG

from finetune_data import get_lm_corpus
#from finetune_models import TransformerSeq
from models import TransformerSeq
from finetune_trainer import train_iteration
from finetune_trainer import full_eval as full_eval_finetune
import datetime
from utils_smoe import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    freeze_gate_weight,
    Logger,
    resize_embedding
)


def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params
):
    # global val
    best_val_loss = None
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]

    if distributed == False or env_params["rank"] == 0:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)

    # DATA
    if data_params['wt103_attack']:
        from data import get_val_test_data
        val_data, test_data = get_val_test_data(data_params, env_params, trainer_params["batch_size"], device, attack = True)
    else:
        
        corpus = get_lm_corpus(data_params["data_path"], data_params["data_name"], attack = data_params['wt103_attack'])
        ntokens = len(corpus.vocab)

        if data_params["data_name"] in ["sst2", "imdb"]:
            num_classes = 2
        elif data_params["data_name"] == "sst5":
            num_classes = 5
        elif data_params["data_name"] == "banking77":
            num_classes = 77

        #eval_batch_size = 10
        eval_batch_size = trainer_params["batch_size"] # set eval batch size to same as train
         # only have train set when not doing word attack
        train_data = corpus.get_iterator("train", trainer_params["batch_size"])
        val_data = corpus.get_iterator("valid", eval_batch_size)
        test_data = val_data


    # MODEL data_params['vocab_size']
    if not data_params['wt103_attack']:
        model = TransformerSeq(
            vocab_size=ntokens,
            **model_params,
            num_classes=num_classes,
            adapt_span_params=adapt_span_params,
            finetune = True
        )
    else:
        from models import TransformerSeq as wikitext_model
        ntokens = 267735 # size of wikitext training corpus 
        model = wikitext_model(
            vocab_size=ntokens,
            **model_params,
            adapt_span_params=adapt_span_params,
        )
    #print(model)
    # if distributed:
    #     local_rank = env_params["local_rank"]
    #     model = model.to(device)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[local_rank],
    #         output_device=local_rank,
    #         find_unused_parameters=True,
    #     )
    # else:
    #     model = torch.nn.DataParallel(model)
    #     model = model.to(device)

    # # OPTIMIZER AND SCHEDULER
    # optimizer, scheduler = get_optimizer_and_scheduler(
    #     model=model, optim_params=optim_params
    # )

    # create logger
    logger = Logger()
    fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
    logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")
    # log paramters
    logging(f"Training Parameters:\n {trainer_params}")
    # logging time
    current_time = datetime.datetime.now()
    logging(str(current_time))
    # log model
    logging(str(model))
    logging(f"Total of Parameters: {sum(p.numel() for p in model.parameters())}")
    logging(
        f"Total of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    # load check points

    logging("=" * 100)
    logging(
        "==== loading pretrained model from {} ====".format(
            trainer_params["pretrained_weight"]
        )
    )
    logging("=" * 100)

    # Load the best saved model.
    # if not trainer_params["full_eval_mode"] or data_params['wt103_attack']: # either load model for training or load for textattack
    #     with open(trainer_params["pretrained_weight"], "rb") as f:
    #         pretrained_model = torch.load(f)
    #     # pdb.set_trace()
    #     pretrained_model_checkpoint = pretrained_model["model"]  # .state_dict()
    #     filtered_checkpoint = {}
    #     #breakpoint()
    #     ### old code ###
    #     for key in pretrained_model_checkpoint.keys():
    #         if not key in model.state_dict():
    #             logging("Can not load {}".format(key))
    #         elif (
    #             not pretrained_model_checkpoint[key].shape
    #             == model.state_dict()[key].shape
    #         ):
    #             logging("Can not load {}, shape do not match".format(key))
    #         else:
    #             filtered_checkpoint[key] = pretrained_model_checkpoint[key]

        ##### DO NOT USE - FROM AN OLDER IMPLEMENTATION
        # # changing the in_emb to match the new dataset
        # pretrained_vocab_size, pretrained_embedding_dim = pretrained_model_checkpoint['module.in_emb.weight'].size()

        # #new_vocab_size = ntokens
        # #model.module.in_emb.weight.data = resize_embedding(pretrained_model_checkpoint['module.in_emb.weight'], new_vocab_size, 'in')
        # #model.module.out_emb.weight.data = resize_embedding(pretrained_model_checkpoint['module.out_emb.weight'], pretrained_vocab_size, 'out')
        
        # model.module.out_emb = resize_embedding(pretrained_model_checkpoint['module.out_emb.weight']).to(device) # out embedding to the old pretrained vocab size
        # model.module.out_proj = torch.nn.Linear(pretrained_vocab_size, model_params['hidden_size']).to(device) # out projection from the old vocab size to the hidden size

        # del pretrained_model_checkpoint['module.in_emb.weight'] # remove the old in-embedding
        #model.load_state_dict(pretrained_model_checkpoint, strict = False)
        ###### DO NOT USE - FROM AN OLDER INPLEMENTATION

    #     model.load_state_dict(filtered_checkpoint, strict=False) # old method
    #     iter_init = 0
    # else:
    #     # resume training from last checkpoint if exists
    #     iter_init = load_checkpoint(
    #         trainer_params["checkpoint_path"],
    #         model,
    #         optimizer,
    #         scheduler,
    #         logger,
    #         distributed,
    #     )
    if not trainer_params["full_eval_mode"]:
        # with open(trainer_params["pretrained_weight"], "rb") as f:
        pretrained_model = torch.load(trainer_params["pretrained_weight"], map_location=device)
            # pretrained_model = torch.load(f)
        # pdb.set_trace()
        # print(pretrained_model["model"].keys())
        # print(model.state_dict())
        pretrained_model_checkpoint = pretrained_model["model"]  # .state_dict()
        # del pretrained_model_checkpoint["module.in_emb.weight"]
        # del pretrained_model_checkpoint["module.out_emb.weight"]
        # del pretrained_model_checkpoint["module.out_emb.bias"]
        filtered_checkpoint = {}
        
        for key in pretrained_model_checkpoint.keys():
            if not key[7:] in model.state_dict():
                logging("Can not load {}".format(key))
            elif (
                not pretrained_model_checkpoint[key].shape
                == model.state_dict()[key[7:]].shape
            ):
                logging("Can not load {}, shape do not match".format(key))
            else:
                filtered_checkpoint[key[7:]] = pretrained_model_checkpoint[key]

        if model_params["base_arch"]  == 'glam':
            print('Using unfiltered pretrained model checkpoint')
            filtered_checkpoint = pretrained_model_checkpoint
        model.load_state_dict(filtered_checkpoint, strict=False)
        iter_init = 0
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    else:
        # resume training from last checkpoint if exists
        iter_init = load_checkpoint(
            trainer_params["pretrained_weight"],
            model,
            optimizer,
            scheduler,
            logger,
            distributed,
            resume,
        )
        print(list(model.parameters()))
        
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        
    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # fix gate
    if model_params["smoe_dropout"]:
        freeze_gate_weight(model)
    # calculate time
    start_time = time.time()
    # eval model
    if trainer_params["full_eval_mode"] and not data_params["wt103_attack"]: # normal full-eval-mode
        # evaluate the model on test data
        with torch.no_grad():
            loss_val, acc_val = full_eval_finetune(
                model,
                optimizer,
                scheduler,
                val_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            loss_test, acc_test = full_eval_finetune(
                model,
                optimizer,
                scheduler,
                test_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if env_params["rank"] == 0:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                else:
                    return

            # log accuracy score
            logging("Val: {:.3f} Acc".format(acc_val))
            logging("Test: {:.3f} Acc".format(acc_test))

        return


    if trainer_params["full_eval_mode"] and data_params["wt103_attack"]: # full eval on text attack
        # evaluate the model on test data
        from trainer import full_eval
    
        with torch.no_grad():
            loss_val, _ = full_eval(
                model,
                optimizer,
                scheduler,
                val_data,
                model_params["block_size"],
                model_params["hidden_size"]
            )
            loss_test, _ = full_eval(
                model,
                optimizer,
                scheduler,
                test_data,
                model_params["block_size"],
                model_params["hidden_size"]
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if env_params["rank"] == 0:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                else:
                    return
                
            logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
            logging("Test: {:.3f} PPL".format(math.exp(loss_test)))

        return

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [
        [
            torch.zeros(
                train_data.bsz,
                model.module.layers[layer_i].attn.attn.get_cache_size(),
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(model.module.attn_layer_count)
        ]
        for _ in range(2)
    ]

    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    for iter_no in range(iter_init, trainer_params["nb_iter"]):
        t_sta = time.time()
        loss_train, acc_train, data_pos[0], hid_cache[0] = train_iteration(
            model,
            model_params["load_balance"],
            optimizer,
            scheduler,
            train_data,
            nb_batches_per_iter,
            model_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            trainer_params["checkpoint_path"]
        )
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, acc_val, data_pos[1], hid_cache[1] = train_iteration(
                model,
                model_params["load_balance"],
                optimizer,
                scheduler,
                val_data,
                nb_batches_per_iter,
                model_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                trainer_params["checkpoint_path"],
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue
        logging(f"=================== EPOCHS {iter_no} ======================")
        # if  ('enwik8' in data_params['data_path']) or ('text8' in data_params['data_path']):
        msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} Acc | loss_val: {:.3f} ~ {:.3f} Acc | elapsed: {:.1f}".format(
            iter_no, loss_train, acc_train, loss_val, acc_val, elapsed
        )
        logging(msg_result)
        # Save the model if the validation loss is the best we've seen so far.
        if (best_val_loss is None) or loss_val < best_val_loss:
            best_val_loss = loss_val
            save_checkpoint(
                trainer_params["checkpoint_path"],
                iter_no,
                model,
                optimizer,
                scheduler,
                logger,
            )
        # save_checkpoint(trainer_params['checkpoint_path'], nb_batches_per_iter, model, optimizer, scheduler, logger)
    end_time = time.time()
    logging(f"Training time total: {(end_time - start_time)/3600} h")


if __name__ == "__main__":
    launch(**get_params(params_config=PARAMS_CONFIG))

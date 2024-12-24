import torch
import tqdm
import time
import math

def evaluate(model, eval_iter, hid_cache):
    skip_first = False # TODO add option to skip first, so far default is False in all other experiments run
    print(f"old behavior: {skip_first}")
    # Turn on evaluation mode which disables dropout.
    model.eval()
    
    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        with tqdm(total=len(eval_iter)) as progress_bar:
            for idx, (data, target, _) in enumerate(eval_iter):
                breakpoint()
                ret, h_cache, gate_top_k_idx = model(data, hid_cache)
                bsz = target.size(1)
                loss, mems = ret[0], ret[1:]
                if skip_first:
                    # TODO add option in the model to skip computing softmax
                    # for all but the last position.
                    # shape of loss: (len,B)
                    # take only the last position, be careful with
                    # proj_adaptive_softmax which can change the indices
                    # if softmax_keep_order is False.
                    loss = loss[-1].sum(dim=-1)  # mean across batch dim
                    total_loss += loss.item()
                    total_len += bsz
                else:
                    if idx == 0:
                        total_len += loss.shape[0] * loss.shape[1]
                        loss = loss.sum()
                        total_loss += loss.item()
                    else:
                        # TODO add option in the model to skip computing
                        # softmax for all but the last position.
                        # shape of loss: (len,B)
                        # take only the last position, be careful with
                        # proj_adaptive_softmax which can change the indices
                        # if softmax_keep_order is False.
                        loss = loss[-1].sum(dim=-1)  # mean across batch dim
                        total_loss += loss.item()
                        total_len += bsz
                progress_bar.update(1)
        total_time = time.time() - start_time
    print(f'{total_len} positions evaluated.')
    print(f'Time : {total_time :.2f}s, '
            f'{ 1000 * total_time / (idx+1):.2f}ms/segment')
    return total_loss / total_len

def format_log(loss, split, dataset = 'wt103'):
    if dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


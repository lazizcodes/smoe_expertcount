import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tree

import pdb
import numpy as np
from fmoe.gates.base_gate import BaseGate

__all__ = [
    "CustomNaiveGate_Balance_SMoE",
    "CustomNaiveGate_Balance_XMoE",
    "CustomNaiveGate_Balance_StableMoE",
    "CustomNaiveGate_Balance_EllipticalXMoE",
    "CustomNaiveGate_Balance_SparseProjectMoE",
    "SpectralGate_SMoE",
    "Balance_Elliptical2XMoE",
    "KSpectral_Balance_SMoE",
    "COSAGate_Balance"
]

class CustomNaiveGate_Balance_SparseProjectMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0
        self.delta = nn.Parameter(torch.tensor(150.))

        #expert_embeddings = torch.empty(num_expert, 8)
        expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False) # for smoe-m, hidden_size = 352, num_heads = 8, therefore head_dim = 44 (double check)
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, gate_top_k_idx, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
        #breakpoint()
        gate = self._sparse_route(reduced_inp, self.expert_embeddings, gate_top_k_idx[:,0])
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
    
    def _sparse_project(self, mat1, cluster_labels, delta = None, stab = 1e-3):
        with torch.no_grad():
            #breakpoint()
            mat1 = mat1.detach()
            cluster_labels = cluster_labels.detach()

            # downsampling trial
            random_indices = torch.randperm(mat1.shape[0])[:2000] # 2000 arbitrarily chosen
            mat1 = mat1[random_indices]
            cluster_labels = cluster_labels[random_indices]

            n, d = mat1.shape
            assert cluster_labels.numel() == n
            
            n_clusters = torch.unique(cluster_labels).numel()
            ##### original, full batch code #####
            # Compute the first term
            diff_matrix = torch.abs(mat1.unsqueeze(1) - mat1.unsqueeze(0))
            first_term = diff_matrix.sum(dim=(0, 1)) / n

            # Compute the second term
            second_term = torch.zeros(d, device=mat1.device)
            for k in range(n_clusters):
                cluster_mask = (cluster_labels == k)
                n_k = cluster_mask.sum()

                if n_k > 0:
                    X_k = mat1[cluster_mask]
                    diff_matrix_k = torch.abs(X_k.unsqueeze(1) - X_k.unsqueeze(0))
                    second_term += diff_matrix_k.sum(dim=(0, 1)) / n_k

            # Compute the final result
            a = first_term - second_term
            ###### original full batch code ######


            ###### loop over features memory-saving code ######
            # a = torch.zeros(d).to(mat1.device)
            # for feature in range(d):
                
            #     # Calculate the first term
            #     x_d = mat1[:, feature]
            #     first_term = torch.sum(torch.abs(x_d.unsqueeze(0) - x_d.unsqueeze(1))) / n

            #     # Calculate the second term
            #     second_term = torch.tensor(0.).to(mat1.device)
            #     for k in range(n_clusters): 
            #         cluster_indices = (cluster_labels == k).nonzero(as_tuple=True)[0]
            #         n_k = len(cluster_indices)
            #         if n_k > 0:  # Ensure there is at least one element in the cluster
            #             x_d_cluster = mat1[cluster_indices, feature]
            #             if n_k > 1:  # Ensure there are at least two elements to compute differences
            #                 second_term += torch.sum(torch.abs(x_d_cluster.unsqueeze(0) - x_d_cluster.unsqueeze(1))) / n_k
            #     a[feature] = first_term - second_term
            ###### loop over features memory-saving code #######
            #breakpoint()
            # compute the soft threshold
            if delta is None:
                delta = self.delta
           
        num = torch.sign(torch.relu(a)) * torch.relu(torch.relu(a) - delta)
        denom = torch.norm(num, p=1)
        w = num / (denom + stab) # stab for stability

        #w = torch.sqrt(w) # trial square root the weights as per Tibshirani
        #breakpoint()

        # collect summary stats for w
        num_zeros = torch.sum(w == 0.).item()
        mx, max_index =  torch.max(w, dim = 0)
        mn, min_index = torch.min(w, dim = 0)
        mean = torch.mean(w).item()
        std = torch.std(w).item()
            
        self.sparse_w_stats = (delta, d, num_zeros, (mx.item(), max_index.item()), (mn.item(), min_index.item()), mean, std)

        #w = w / torch.max(w) #### get rid of this hardcoding
        return w
    
    def _sparse_route(self, mat1, mat2, cluster_labels, eps = 1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        #breakpoint()
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps) 
        # mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps) # trial removing expert normalization
        W = self._sparse_project(mat1, cluster_labels)
        mat1W = mat1 * W
        #mat1W = mat1
        return mat1W.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
        
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


class CustomNaiveGate_Balance_EllipticalXMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False, root_invert = False, intra_layer =  False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        #expert_embeddings = torch.empty(num_expert, 8)
        expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False) # for smoe-m, hidden_size = 352, num_heads = 8, therefore head_dim = 44 (double check)
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale
        self.root_invert = root_invert
        self.intra_layer = intra_layer

        self.gate_W = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.gate_W_scaled = (0.0, 0.0, 0.0, 0.0, 0.0)

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, fwds =  None, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
        #breakpoint()
        if type(fwds) is not tuple: # generic xmoe
            gate = self._cosine(reduced_inp, self.expert_embeddings)
        else: # use elliptical once fwds is fully populated with (fwd_last, fwd_last2)
            gate = self._elliptical_cosine(reduced_inp, self.expert_embeddings, fwds)
        #gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
     
    
    def _elliptical_cosine(self, mat1, mat2, fwds, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        #breakpoint()
        v_last, v_last2 = fwds
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps) 
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        W = self.compute_W(v_last, v_last2)
        
        mat1W = mat1 @ W
        #mat1W = mat1

        return mat1W.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
    
    def compute_W(self, v_last, v_last2, k_last = None, k_last2 = None, delta = 1):
        with torch.no_grad():
            # v shape: [(bsize x seqlen) x dim]
            v_last = v_last.detach()
            v_last2 = v_last2.detach()
            # k_last = k_last.detach()
            # k_last2 = k_last2.detach()
            
            if self.intra_layer:
                w =  torch.var(v_last, dim = 0) # shape: [dim]

                if self.root_invert:
                    w = 1 / (w + 1e-4)
                
                # store weights  
                W_mean = torch.mean(w)
                W_std = torch.std(w)
                W_max, max_idx = torch.max(w, dim = 0)
                W_min, min_idx = torch.min(w, dim = 0)
                
                self.gate_W = (w, W_std, W_max, max_idx, W_min, min_idx, W_mean)
                if self.mean_scale:
                    weights_scaled = w / torch.mean(w)
                else:
                    weights_scaled = w / torch.max(w)
                scaled_mean = torch.mean(weights_scaled)
                scaled_std = torch.std(weights_scaled)
                scaled_max = torch.max(weights_scaled)
                scaled_min = torch.min(weights_scaled)
                self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)


                w = w / torch.max(w)

                return torch.diag_embed(w)

                

        #return v_last
            seqlen = v_last.size(0)
            if delta is None:
                deltas = torch.abs(k_last - k_last2) #include small term for stability and gradient attenuation
                difference_quotients = (v_last - v_last2) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            else:
                #delta = torch.mean(torch.abs(k_last - k_last2))
                difference_quotients = (v_last-v_last2) / delta
            
            W = torch.norm(difference_quotients, p = 1, dim = 0) /seqlen #columnwise average l1 norms
            # W dim: [h]

            if self.root_invert:
                #W = 1/ (torch.sqrt(W) + 1e-3) # trial D^{-1/2} as from sphering the data in LDA
                W  = W**2 # trial squaring and inverting
                W = 1 / (W + 1e-4) # trial inversion but no sqrt

            # store weights  
            W_mean = torch.mean(W)
            W_std = torch.std(W)
            W_max, max_idx = torch.max(W, dim = 0)
            W_min, min_idx = torch.min(W, dim = 0)
            
            self.gate_W = (W, W_std, W_max, max_idx, W_min, min_idx, W_mean)
            if self.mean_scale:
                weights_scaled = W / torch.mean(W)
            else:
                weights_scaled = W / torch.max(W)
            scaled_mean = torch.mean(weights_scaled)
            scaled_std = torch.std(weights_scaled)
            scaled_max = torch.max(weights_scaled)
            scaled_min = torch.min(weights_scaled)
            self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)

            
            if self.mean_scale:
                W = W / torch.mean(W)
            else: # default max scale
                #W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale for multidim W
                W = W / torch.max(W) # W is 0-dim here just [h] vector.
                #print('here')
            W = torch.diag_embed(W)
            
        return W
      

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores



class Balance_Elliptical2XMoE(BaseGate): # perform over-layers averaging but of inp(layer) and inp(layer-1) rather than of fwds
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False, root_invert = False, intra_layer  = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        #expert_embeddings = torch.empty(num_expert, 8)
        expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False) # for smoe-m, hidden_size = 352, num_heads = 8, therefore head_dim = 44 (double check)
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale
        self.root_invert = root_invert

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, inp_last =  None, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._elliptical_cosine(inp, self.expert_embeddings, inp_last)
        #gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
     
    
    def _elliptical_cosine(self, inp, expert_embeddings, inp_last =  None, eps = 1e-4):
        assert inp.dim() == 2
        assert expert_embeddings.dim() == 2
        #breakpoint()
        if inp_last is not None:
            W = self.compute_W(inp, inp_last)
            inp = F.normalize(inp, p=2.0, dim=1, eps=eps)
            expert_embeddings = F.normalize(expert_embeddings.float(), p=2.0, dim=1, eps=eps)
            inp = inp * W
            #breakpoint()

        else:
            inp = F.normalize(inp, p=2.0, dim=1, eps=eps)
            expert_embeddings = F.normalize(expert_embeddings.float(), p=2.0, dim=1, eps=eps)
           
        return inp.float().matmul(expert_embeddings.transpose(0, 1)).type_as(inp)
    
    def compute_W(self, inp, inp_last):
        with torch.no_grad():
            # v shape: [(bsize x seqlen) x dim]
            inp = inp.detach()
            inp_last = inp_last.detach()
            # k_last = k_last.detach()
            # k_last2 = k_last2.detach()

        #return v_last
            seqlen = inp.size(0)
            # if delta is None:
            #     deltas = torch.abs(k_last - k_last2) #include small term for stability and gradient attenuation
            #     difference_quotients = (v_last - v_last2) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            #delta = torch.mean(torch.abs(k_last - k_last2))
            difference_quotients = inp-inp_last
            
            W = torch.norm(difference_quotients, p = 1, dim = 0) /seqlen #columnwise average l1 norms
            # W dim: [h]
            
            if self.root_invert:
                W = 1/ (torch.sqrt(W) + 1e-3) # trial D^{-1/2} as from sphering the data in LDA

            # store weights  
            W_mean = torch.mean(W)
            W_std = torch.std(W)
            W_max, max_idx = torch.max(W, dim = 0)
            W_min, min_idx = torch.min(W, dim = 0)
            self.gate_W = (W, W_std, W_max, max_idx, W_min, min_idx, W_mean)
            if self.mean_scale:
                weights_scaled = W / torch.mean(W)
            else:
                weights_scaled = W / torch.max(W)
            scaled_mean = torch.mean(weights_scaled)
            scaled_std = torch.std(weights_scaled)
            scaled_max = torch.max(weights_scaled)
            scaled_min = torch.min(weights_scaled)
            self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)

            #self.gate_deltas =  torch.max(torch.abs(k_last-k_last2)), torch.min(torch.abs(k_last-k_last2)), torch.mean(torch.abs(k_last - k_last2)), torch.std(torch.abs(k_last-k_last2))

            
            if self.mean_scale:
                W = W / torch.mean(W)
            else: # default max scale
                #W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale for multidim W

                W = W / torch.max(W) # W is 0-dim here just [h] vector.
                #print('here')
            #W = torch.diag_embed(W)
            
        return W
      

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores




class SpectralGate_SMoE(BaseGate):
    def __init__(self, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        #self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None
        self.num_expert = num_expert

    def forward(self, attn_logit, return_all_scores=False):

        gate_top_k_val, gate_top_k_idx =  self.spectral_cluster(attn_logit)
        
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # if self.dense_moe_flag:
        #     gate = torch.ones_like(gate)  # average the importance of all experts
        #     gate_top_k_val, gate_top_k_idx = torch.topk(
        #         gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
        #     )
        #     gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        # else:
        #     gate_top_k_val, gate_top_k_idx = torch.topk(
        #         gate, k=self.top_k, dim=-1, largest=True, sorted=False
        #     )  # [.. x top_k]
        #     gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        # gate_score = F.softmax(gate_top_k_val, dim=-1)
        # if self.g_blance:
        #     self.set_load_balance(gate, gate_top_k_idx)

        # if return_all_scores:
        #     return gate_top_k_idx, gate_score, gate

        return gate_top_k_idx, gate_score
    
    def spectral_cluster(self, attn_logit, threshold  = None):
        with torch.no_grad():
            attn_logit =  attn_logit.detach()
            bsize, seqlen, _ =  attn_logit.shape
            if threshold is not None:
                # apply thresholding to sparsify
                attn_logit[attn_logit <= threshold] = 0.
            
            D = torch.eye(seqlen, device = attn_logit.device)*seqlen # degree matrix
            L = D - attn_logit # laplacian
            eigvals, eigvecs = torch.linalg.eigh(L) 
            #breakpoint()
            eigvecs = eigvecs[:, :, :self.num_expert] # arranged in ascending order
            eigvecs =  eigvecs.reshape(bsize*seqlen, -1)
            gate_top_k_val, gate_top_k_idx = torch.topk(eigvecs, k = self.top_k, largest=True, sorted = False)
        
        return gate_top_k_val, gate_top_k_idx






class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomNaiveGate_Balance_XMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, 8)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        reduced_inp = self.inp_reduction(inp)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

class COSAGate_Balance(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        #self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)
        self.avg_distances_per_cluster = 0.0, 0.0, 0.0, 0.0, 0.0
        self.W = 0.0, 0.0, 0.0, 0.0, 0.0

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, gate_top_k_idx, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosa_dot(reduced_inp, self.expert_embeddings, gate_top_k_idx[:,0])
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
    
    def _cosa_dot(self, mat1, mat2, cluster_labels):
        # with torch.no_grad():
        #     mat1 = mat1.detach()
        W = self.compute_W(mat1, cluster_labels)
        assert W.requires_grad == False
        #print('here')

        mat1 = mat1 * W
        
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def compute_W(self, tokens, cluster_assignments, lamb = None):
        with torch.no_grad():
            tokens = tokens.detach()
            cluster_assignments = cluster_assignments.detach()

            # downsample for speedup
            # random_indices = torch.randperm(tokens.shape[0])[:2000] # 2000 arbitrarily chosen
            # tokens = tokens[random_indices] 
            # cluster_assignments_d = cluster_assignments[random_indices]
            cluster_assignments_d = cluster_assignments

            # get [n,n,d] distances tensor
            distances = torch.abs(tokens[:, None, :] - tokens[None, :, :])

            n, _, d = distances.shape
            k = cluster_assignments_d.max().item() + 1

            ### use assumption that any cluster clusters on the same attributes. 
            k = 1

            avg_distances_per_cluster = torch.zeros(d, k).to(tokens.device)
            for cluster_id in range(k):
                cluster_mask = (cluster_assignments_d == cluster_id).float()  # Shape [n]
                cluster_indices = torch.nonzero(cluster_mask).squeeze()  # Get indices of points in this cluster

                if cluster_indices.dim() == 0:  # Only one element in the cluster
                    continue  # Skip as there's no other vector to compare within the cluster

                # Only consider the upper triangular part of the matrix
                upper_tri_mask = torch.triu(torch.ones((n, n)), diagonal=1).bool().to(cluster_mask.device)
                mask = cluster_mask.unsqueeze(1) * cluster_mask.unsqueeze(0) * upper_tri_mask

                masked_distances = distances * mask.unsqueeze(-1)  # Shape [n, n, d]
                total_distances = masked_distances.sum(dim=(0, 1))  # Sum over n, n for each d
                counts = mask.sum(dim=(0, 1))  # Number of valid pairs in the cluster

                counts = torch.where(counts == 0, torch.tensor(1.0), counts)  # Avoid division by zero
                avg_distances_per_cluster[:, cluster_id] = total_distances / counts

            # get exponential distances
            if lamb is None:
                #lamb = torch.sqrt(torch.tensor(d))
                lamb = torch.tensor(1.)

            #self.avg_distances_per_cluster = avg_distances_per_cluster[:20], avg_distances_per_cluster.min(), avg_distances_per_cluster.max(), avg_distances_per_cluster.mean(), avg_distances_per_cluster.std() 

            W = F.softmax(-avg_distances_per_cluster / lamb, dim = 0) # colwise softmax
            W = W.squeeze() # accomodate [d,1] shape 
            self.W = W[:20], W.min(), W.max(), W.mean(), W.std()
            #breakpoint()
            # broadcast up and transpose to shape [n, d] which rows arranged by corresponding cluster assignments of n using original cluster assignments
            #W = W[:, cluster_assignments].T
            

        return W

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores



class CustomNaiveGate_Balance_StableMoE(BaseGate):
    r"""
    Naive Gate StableMoE
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self._cosine(inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

class KSpectral_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False, gate_then_mix = True, gate_with_eigenvectors = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.eigengate = nn.Linear(self.tot_expert, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

        self.gate_then_mix = gate_then_mix
        self.gate_with_eigenvectors = gate_with_eigenvectors

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, eigenvectors, return_all_scores=False):
        assert (self.gate_then_mix != self.gate_with_eigenvectors) # exactly one has to be true
        if self.gate_then_mix:
            gate = self.gate(inp) # d_model -> num_exp
            gate = self.mix_embedding(gate, eigenvectors)
        
        elif self.gate_with_eigenvectors:
            eigenvectors = eigenvectors.detach()
            assert eigenvectors.requires_grad == False
            B, _, M = eigenvectors.shape
            eigenvectors = eigenvectors.reshape(B*M, -1)
            gate = self.eigengate(eigenvectors)



        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
    
    def mix_embedding(self, gate, eigenvectors):
        # mix gate output with its own spectral embedding
        # eigvectors:  B x num_exp x M
        # gate: B_M x num_exp
        with torch.no_grad():
            eigenvectors = eigenvectors.detach().clone()
            gate = gate.detach().clone()
            B, _, M = eigenvectors.shape

            eigenvectors = eigenvectors.reshape(B*M, -1)
            # add and normlalize
            out = gate + eigenvectors 
            out = torch.nn.functional.normalize(out, p = 2)
        return out



import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from custom_transformer import FMoETransformerMLP, FMoETransformerMLPOpt
from custom_gates import *


# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span
def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, hidden_size, attn_span, dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params["adapt_span_enabled"]
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(
                attn_span=attn_span, **adapt_span_params, **kargs
            )

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe
            )

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class EllipticalAttention(nn.Module):
    def __init__(self, hidden_size, attn_span, M, dropout, adapt_span_params, show_M = False, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params["adapt_span_enabled"]
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(
                attn_span=attn_span, **adapt_span_params, **kargs
            )
        self.M = M
        self.show_M = show_M

    def W_over_layers(self, value, value_last, key = None, key_last = None, delta = None, attenuation = 1e-3):
        with torch.no_grad():
            value = value.detach()
            value_last = value_last.detach()
            # key = key.detach()
            # key_last = key_last.detach()

            #v dim: [bsz x (m + l) x h]

            seqlen = value.size(1)
            if delta is None:
                deltas = torch.abs(key - key_last) + attenuation #include small term for stability and gradient attenuation
                difference_quotients = (value - value_last) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            else:
                #delta = torch.mean(torch.abs(key - key_last))
                difference_quotients = (value-value_last) / delta
            
            # difference quotients dim: [bsz x (m+l) x h]
            W = torch.norm(difference_quotients, p = 1, dim = 1) /seqlen #columnwise average l1 norms, [bsize x nhead x dhead]
            # W dim: [bsz x h]

            if self.show_M:
                weights = W[0] #first batch
                W_std = torch.std(weights)
                self.W = (weights, W_std)

                weights_scaled = weights / torch.max(weights)
                scaled_std = torch.std(weights_scaled)
                self.W_scaled = (weights_scaled, scaled_std)

                self.deltas =  torch.max(torch.abs(key-key_last)), torch.min(torch.abs(key-key_last)), torch.mean(torch.abs(key - key_last)), torch.std(torch.abs(key-key_last))

            
            W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale
            W = torch.diag_embed(W)
            
        return W
    
    def forward(self, query, key, value, key_pe, key_last = None, value_last = None):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H
        #breakpoint()

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe
            )

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        if self.M:
            W = self.W_over_layers(value, value_last, delta = 1)
            query = query @ W
            #print('here')
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos
        
        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        
        return out, key, value
        

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span



class MultiHeadEllipticalAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, M = False, show_M = False, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = EllipticalAttention(hidden_size=self.head_dim, nb_heads=nb_heads, M = M, show_M = show_M, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        self.M = M

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe, key_last = None, value_last = None):
        #breakpoint()
        
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)


        out, key, value = self.attn(query, key, value, key_pe, key_last, value_last)  # B_K x M x D
        
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        
        return out, key, value

class SeqSpectralAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, hidden_size, attn_span, dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params["adapt_span_enabled"]
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(
                attn_span=attn_span, **adapt_span_params, **kargs
            )


    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe
            )

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L
        #breakpoint() #TODO clarify what to do about the M x L shape, also  clarify the zeros in the key - ANS: take final M
        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn_logit = attn_cont + attn_pos

        attn = attn_logit / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        return out, attn_logit[:, :, :query.size(1)] # take only BxMxM entries from the logits

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span



class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, hidden_size, attn_span, dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params["adapt_span_enabled"]
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(
                attn_span=attn_span, **adapt_span_params, **kargs
            )

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe
            )

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span

class MultiHeadSeqSpectralAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqSpectralAttention(hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out, attn_logit = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        attn_logit = self.head_combine(attn_logit.view(B, K, M, M).detach(), self.proj_out.weight.detach())
        
        return out, attn_logit

    def head_combine(self, attn_logit, proj_weights):
        with torch.no_grad():
            attn_logit.detach()
            proj_weights = proj_weights.detach()
            #breakpoint()
            proj_weights = proj_weights.view(self.nb_heads, self.head_dim, -1) # [K x D x KD]
            block_norms = torch.norm(proj_weights, p = 'fro', dim  = (1,2))
            head_combine_weights = (block_norms / block_norms.sum()).view(1, self.nb_heads, 1, 1) # broadcast up

            attn_logit = (attn_logit  * head_combine_weights).sum(dim  =  1) # combine over heads dimension
        return attn_logit



class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, kspectral = False, moe_num_experts = 16, **kargs): # moe_num_experts taken from CustomizedMoEPositionwiseFF init setting
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kspectral = kspectral
        self.moe_num_experts = moe_num_experts

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        if self.kspectral:
            val = value
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D   
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        if self.kspectral:
            eigvectors = self.select_eigvectors(out, val, self.moe_num_experts)
        out = self.proj_out(out)
        if self.kspectral:
            return out, eigvectors
        return out

    def select_eigvectors(self, out, value, num_experts):
        with torch.no_grad():
            out = out.detach() # B x M x K_D
            value = value.detach() # B x (M+L) x K_D
            M = out.size(1)
            # componentwise divide out by value
            eigcols = out / value[:, -M:, :] # taking final M tokens in sequence
            # B x D
            eigcols = torch.norm(eigcols, dim = 1) # columnwise norms
            idxs = torch.topk(eigcols, dim = -1, k = num_experts, largest=True, sorted = False)[1]
            
            X = value[:, -M:, :].permute(0,2,1) # B x K_D x M
            eigvectors = torch.gather(X, dim=1, index=idxs.unsqueeze(-1).expand(-1, -1, X.shape[-1])) # B x num_exp x M
        return eigvectors




class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class SpectralMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
        layerth=0
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
            spectral_gate=True)
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layerth = layerth

    def forward(self, inp, attn_logit):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            #breakpoint()
            ##### positionwise feed-forward
            core_out, gate_top_k_idx = super().forward(inp, attn_logit = attn_logit)

            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output, gate_top_k_idx


class CustomizedMoEPositionwiseFF(FMoETransformerMLP): 
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
        layerth=0,
        elliptical_gate=False,
        elliptical_gate2 = False,
        kspectral_gate = False,
        show_gate_W = False,
        mean_scale = False,
        root_invert = False,
        intra_layer = False,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
            elliptical_gate=elliptical_gate,
            elliptical_gate2=elliptical_gate2,
            kspectral_gate=kspectral_gate,
            show_gate_W = show_gate_W,
            mean_scale = mean_scale,
            root_invert = root_invert,
            intra_layer =  intra_layer
        )
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layerth = layerth

    def forward(self, inp, gate_top_k_idx = None, fwds =  None, moe_inp_last = None, eigenvectors = None):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            if self.elliptical_gate:
                core_out, gate_top_k_idx, fwds = super().forward(inp, gate_top_k_idx, fwds)
            elif self.elliptical_gate2:
                core_out, gate_top_k_idx, moe_inp_last = super().forward(inp, moe_inp_last=moe_inp_last)
            elif self.kspectral_gate:
                core_out, gate_top_k_idx = super().forward(inp, eigenvectors=eigenvectors)
            else: #sparseproj, cosa
                core_out, gate_top_k_idx = super().forward(inp, gate_top_k_idx)


            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)
        if self.elliptical_gate:
            return output, gate_top_k_idx, fwds
        elif self.elliptical_gate2:
            return output, gate_top_k_idx, moe_inp_last
        else:
            return output, gate_top_k_idx,

class CustomizedMoEPositionwiseFFMoM(FMoETransformerMLP):
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
        gamma=1.0,
        mu=0.9,
        beta1=0.9,
        beta2=0.999,
        layerth=0
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
            layerth=layerth
        )
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.layerth = layerth


    def forward(self, inp, moment):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            # core_out, gate_top_k_idx = super().forward(inp)
            # core_out = self.dropout(core_out)

            # if self.layerth==0:
            if True:
                ##### ADAM
                # epsilon = 0.05 * core_out / torch.norm(core_out, keepdim=True)
                # core_out, gate_top_k_idx = super().forward(inp + epsilon)
                # momentum = self.mu * moment[2] + core_out
                # p = moment[0]
                # v = moment[1]
                # p = self.beta1 * p + (1-self.beta1) * core_out
                # v = self.beta2 * v + (1-self.beta2) * (core_out ** 2)
                # # adam = (self.gamma / torch.sqrt((v/(1-self.beta2**(1+self.layerth))+1e-8))) * (p / (1-self.beta1**(1+self.layerth)))
                # adam = (self.gamma / torch.sqrt(v+1e-8)) * p + inp

                ##### Robust momentum
                L = 1.0
                eps = self.mu ** 3 / ((self.gamma-1)*((1-self.mu)**2)*(1+self.mu))
                beta = self.gamma*((1-self.mu)**2)*(1+self.mu)/L
                alpha = self.gamma*(self.mu**3)/(self.gamma-1)
                new_inp = inp + eps * beta * moment
                core_out, gate_top_k_idx = super().forward(new_inp)
                moment = - core_out + alpha * moment

                ##### residual connection + layer normalization
                # output = self.layer_norm(inp - adam)
                output = self.layer_norm(inp + beta * moment)
            
            # else:
                ##### Momentum
                # epsilon = 0.05 * core_out / torch.norm(core_out, keepdim=True)
                # core_out, gate_top_k_idx = super().forward(inp + epsilon)
                ##### Adam
                # p = moment[0]
                # v = moment[1]
                # momentum = self.mu * moment[2] + core_out
                # output = self.layer_norm(inp - momentum)

        # return output, (p,v,momentum), gate_top_k_idx
        return output, moment, gate_top_k_idx

class CustomizedMoEPositionwiseFFOpt(FMoETransformerMLPOpt):
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
            freq=freq,
            alpha=alpha,
            act_experts=act_experts,
            g_blance=g_blance,
            opt_blance=opt_blance,
            combine_gate=combine_gate,
            opt_loss=opt_loss,
        )
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class TransformerSeqLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        s,
        g,
        f,
        gate_name,
        optimal_policy,
        moe_top_k,
        freq,
        alpha,
        act_experts,
        g_blance,
        opt_blance,
        combine_gate,
        opt_loss,
        gamma,
        mu,
        layerth,
        compute_rep_collapse = False,
        show_gate_W = False,
        mean_scale = False,
        root_invert = False,
        intra_layer = False,
        **kargs,
    ):
        nn.Module.__init__(self)
        ### warning: the gate_name here will implement the gate for layers NOT using whatever custom layer we're testing. Hence if we're using a regular smoe on the first layer (g), then gate_name needs to be smoe
        if gate_name in ["smoe", "smoe-dropout"]:
            gate = CustomNaiveGate_Balance_SMoE
        elif gate_name == "xmoe":
            gate = CustomNaiveGate_Balance_XMoE
        elif gate_name == 'ellipticalmoe': # actually redundant in code, in the self.smoe layer i pass the gate straight into the FFMOE
            gate = CustomNaiveGate_Balance_EllipticalXMoE
        elif gate_name == 'elliptical2':
            gate = Balance_Elliptical2XMoE
        elif gate_name == "stablemoe":
            gate = CustomNaiveGate_Balance_StableMoE
        elif gate_name == 'spectralmoe': # actually redundant in code, in the self.smoe layer i pass the gate straight into the FFMOE
            gate = SpectralGate_SMoE
        elif gate_name == 'sparseproject': # actually redundant in code, in the self.smoe layer i pass the gate straight into the FFMOE
            gate = CustomNaiveGate_Balance_SparseProjectMoE
        elif gate_name == 'cosa': # actually redundant in code, in the self.smoe layer i pass the gate straight into the FFMOE
            gate = COSAGate_Balance
        else:
            print(f"{gate_name} has not been implemented yet!")

        self.attn = (
            MultiHeadSeqAttention(hidden_size = hidden_size, dropout = dropout, **kargs)
            if s is 's'
            else MultiHeadEllipticalAttention(hidden_size = hidden_size, dropout = dropout, M = False, **kargs)
            if s is "r"
            else MultiHeadEllipticalAttention(hidden_size = hidden_size, dropout = dropout, M = True, **kargs)
            if s is "e"
            else MultiHeadSeqSpectralAttention(hidden_size = hidden_size, dropout = dropout, **kargs)
            if s is  "q"
            else MultiHeadSeqAttention(hidden_size = hidden_size, dropout=dropout, kspectral = True, **kargs)
            if s is "k"
            else None
        )
        if optimal_policy: # always false
            self.smoe = (
                CustomizedMoEPositionwiseFFOpt(
                    gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    freq=freq,
                    alpha=alpha,
                    act_experts=act_experts,
                    g_blance=g_blance,
                    opt_blance=opt_blance,
                    combine_gate=combine_gate,
                    opt_loss=opt_loss,
                )
                if g is "g"
                else None
            )
        else: 
            self.smoe = (
                CustomizedMoEPositionwiseFF(
                    CustomNaiveGate_Balance_EllipticalXMoE,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth,
                    elliptical_gate=True,
                    show_gate_W=show_gate_W,
                    mean_scale=mean_scale,
                    root_invert = root_invert,
                    intra_layer = intra_layer
                )
                if g is "E"
                else
                CustomizedMoEPositionwiseFF(
                    Balance_Elliptical2XMoE,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth,
                    elliptical_gate2=True,
                    show_gate_W=show_gate_W,
                    mean_scale=mean_scale,
                    root_invert = root_invert
                )
                if g is "o"
                else
                CustomizedMoEPositionwiseFF(
                    CustomNaiveGate_Balance_SparseProjectMoE,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth
                )
                if g is "p"
                else
                CustomizedMoEPositionwiseFF(
                    COSAGate_Balance,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth
                )
                if g is "c"
                else
                CustomizedMoEPositionwiseFF(
                    gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth
                )
                if g is "g"
                else 
                CustomizedMoEPositionwiseFFMoM(
                    gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    gamma=gamma,
                    mu=mu,
                    layerth=layerth,
                )
                if g is "m"
                else 
                CustomizedMoEPositionwiseFF(
                    KSpectral_Balance_SMoE,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth,
                    kspectral_gate=True
                )
                if g is "k"
                else 
                SpectralMoEPositionwiseFF(
                    SpectralGate_SMoE,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    layerth=layerth,
                )
                if g is "Q"
                else None
            )

        self.ff = (
            FeedForwardLayer(
                hidden_size=hidden_size,
                inner_hidden_size=inner_hidden_size,
                dropout=dropout,
            )
            if f is "f"
            else None
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.use_attn = s == "s" or s == "e" or s == 'q' or s == 'k' or s == 'r'
        self.use_smoe = g == "g" or g == "m" or g == 'E' or g == 'p' or g == 'Q' or g == 'o' or g == 'k' or g =='c'
        self.g = g
        self.s = s
        self.use_ff = f == "f"

        self.compute_rep_collapse = compute_rep_collapse

    def forward(self, h, h_cache, moment, key_pe, gate_top_k_idx = None, fwds = None, moe_inp_last = None):
        # h = B x M x H
        # h_cache = B x L x H
        #breakpoint()
        if type(h) is tuple: 
            h, key_last, value_last = h
        if self.use_attn:
            h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
            if self.s == 'e': # elliptical
                attn_out, key_last, value_last = self.attn(h, h_all, h_all, key_pe, key_last, value_last)
            elif self.s == 'q': # spectral
                attn_out, attn_logit = self.attn(h, h_all, h_all, key_pe)
            elif self.s == 'p':
                attn_out, eigenvectors = self.attn(h, h_all, h_all, key_pe)
            elif self.s =='r': # pass forward key and query but no elliptical comp
                attn_out, key_last, value_last = self.attn(h, h_all, h_all, key_pe)
            else: # all other attentions
                attn_out = self.attn(h, h_all, h_all, key_pe)
            h = self.norm1(h + attn_out)  # B x M x H
        if self.use_smoe:
            if self.g == "m": # momentum
                smoe_out, moment, gate_top_k_idx, fwds = self.smoe(h, moment, fwds)
            if self.g == "g": # standard gate specified argparse
                smoe_out, gate_top_k_idx = self.smoe(h, gate_top_k_idx)
            if self.g == "E": # elliptical gate
                smoe_out, gate_top_k_idx, fwds = self.smoe(h, gate_top_k_idx, fwds)
            if self.g == 'o': # elliptical2 gate
                smoe_out, gate_top_k_idx, moe_inp_last = self.smoe(h, moe_inp_last = moe_inp_last)
            if self.g == 'p' or self.g == 'c': # sparse project or cosa
                smoe_out, gate_top_k_idx = self.smoe(h, gate_top_k_idx) 
            if self.g == 'Q': # spectral gate
                smoe_out, gate_top_k_idx = self.smoe(h, attn_logit)
            if self.g == 'k': #kpca-spectral
                smoe_out, gate_top_k_idx = self.smoe(h, eigenvectors=eigenvectors)
            if self.compute_rep_collapse:
                self.rep_collapse(smoe_out)
            h = self.norm2(h + smoe_out)  # B x M x H
        if self.use_ff:
            ff_out = self.ff(h)
            h = self.norm3(h + ff_out)  # B x M x H
        if self.s == 'e' or self.s == 'r': # repackage out, key, value in elliptical attention
            h = h, key_last, value_last
        
        if self.g == "E":
            return h, moment, gate_top_k_idx, fwds
        elif self.g == 'o':
            return h, moment, gate_top_k_idx, moe_inp_last
        else:
            return h, moment, gate_top_k_idx
        
        return h, moment, None, fwds # feed forward layer returns no gate indices, but layer output expects four-tuple

    def rep_collapse(self, x):
        #breakpoint()
        n = x.shape[1]
        x_norm = torch.norm(x, 2, dim = -1, keepdim = True)
        x_ = x / x_norm
        x_cossim = torch.tril((x_ @ x_.transpose(-2,-1 )), diagonal = -1).sum(dim = (-1,-2)) / (n*(n-1)/2)
        x_cossim = x_cossim.mean()
        self.cossim = x_cossim

        return

class TransformerSeq(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_hidden_size,
        nb_heads,
        nb_layers,
        attn_span,
        architecture,
        base_arch,
        gate_name,
        optimal_policy,
        dropout,
        moe_top_k,
        freq,
        alpha,
        act_experts,
        g_blance,
        opt_blance,
        combine_gate,
        opt_loss,
        gamma,
        mu,
        layer_n,
        compute_rep_collapse = False,
        show_gate_W = False,
        mean_scale = False,
        root_invert = False,
        intra_layer = False,
        num_experts = None,
        num_classes = None,
        finetune = False,
        **kargs,
    ):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        if finetune:
            self.out_emb = nn.Linear(hidden_size, hidden_size)
        else:
            self.out_emb = nn.Linear(hidden_size, vocab_size)
        # position embeddings
        self.key_pe = nn.Parameter(torch.randn(1, hidden_size // nb_heads, attn_span))
        if finetune:
            assert num_classes is not None
            self.project_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes),
            )

        self.finetune = finetune
        arch = architecture
        self.arch = arch
        self.layer_n = layer_n
        print(arch)
        self.attn_layer_count = arch.count("s") +  arch.count("e")  + arch.count('q') + arch.count('k') + arch.count("r")
        self.layers = nn.ModuleList()
        if base_arch == "transformer":
            self.layers.extend(
                TransformerSeqLayer(
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    s=arch[2 * i],
                    g=arch[2 * i + 1],
                    f=None,
                    gate_name=gate_name,
                    optimal_policy=optimal_policy,
                    nb_heads=nb_heads,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    freq=freq,
                    alpha=alpha,
                    act_experts=act_experts,
                    g_blance=g_blance,
                    opt_blance=opt_blance,
                    combine_gate=combine_gate,
                    opt_loss=opt_loss,
                    attn_span=attn_span,
                    gamma=gamma,
                    mu=mu,
                    layerth=i,
                    compute_rep_collapse= compute_rep_collapse,
                    show_gate_W= show_gate_W,
                    mean_scale = mean_scale,
                    root_invert = root_invert,
                    intra_layer= intra_layer,
                    num_experts=num_experts,
                    **kargs,
                )
                for i in range(nb_layers)
            )
        elif base_arch == "glam":
            for i in range(nb_layers):
                self.layers.extend(
                    [
                        TransformerSeqLayer(
                            hidden_size=hidden_size,
                            inner_hidden_size=inner_hidden_size,
                            s=arch[2 * i],
                            g=arch[2 * i + 1],
                            f=None,
                            gate_name=gate_name,
                            optimal_policy=optimal_policy,
                            nb_heads=nb_heads,
                            dropout=dropout,
                            moe_top_k=moe_top_k,
                            freq=freq,
                            alpha=alpha,
                            act_experts=act_experts,
                            g_blance=g_blance,
                            opt_blance=opt_blance,
                            combine_gate=combine_gate,
                            opt_loss=opt_loss,
                            attn_span=attn_span,
                            gamma = gamma,
                            mu = mu,
                            layerth = i,
                            compute_rep_collapse= compute_rep_collapse,
                            show_gate_W= show_gate_W,
                            mean_scale = mean_scale,
                            intra_layer = intra_layer,
                            **kargs,
                        ),
                        TransformerSeqLayer(
                            hidden_size=hidden_size,
                            inner_hidden_size=inner_hidden_size,
                            s=arch[2 * (i + 1)],
                            g=None,
                            f=arch[2 * (i + 1) + 1],
                            gate_name=gate_name,
                            optimal_policy=optimal_policy,
                            nb_heads=nb_heads,
                            dropout=dropout,
                            moe_top_k=moe_top_k,
                            freq=freq,
                            alpha=alpha,
                            act_experts=act_experts,
                            g_blance=g_blance,
                            opt_blance=opt_blance,
                            combine_gate=combine_gate,
                            opt_loss=opt_loss,
                            attn_span=attn_span,
                            gamma = gamma,
                            mu = mu,
                            layerth = i,
                            compute_rep_collapse= compute_rep_collapse,
                            show_gate_W= show_gate_W,
                            mean_scale = mean_scale,
                            intra_layer = intra_layer,
                            **kargs,
                        ),
                    ]
                )

        else:
            raise RuntimeError(
                "wrong type of base architecture - must be 'transformer' or 'glam'"
            )

    def forward(self, x, h_cache):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        h_cache_next = []
        moment = torch.zeros_like(h)
        gate_top_k_idx_out = None
        for l, layer in enumerate(self.layers):
            #breakpoint()
            if layer.use_attn:
                cache_size = layer.attn.attn.get_cache_size() # cache_size = attention_span, 1024 for m
                if type(h) is tuple:
                    h, key_last, value_last = h
                if cache_size > block_size:
                    h_cache_next_l = torch.cat(
                        [h_cache[l][:, -cache_size + block_size :, :], h], dim=1
                    ).detach()
                else:
                    h_cache_next_l = h[:, -cache_size:, :].detach()
                h_cache_next.append(h_cache_next_l)
                if layer.s == 'e': # elliptical attention
                    h = h, key_last, value_last

                ### get this logic working. 
                if layer.g == 'E': # elliptical gate
                    try:
                        h, moment, gate_top_k_idx, fwds = layer(h, h_cache[l], moment, self.key_pe, fwds = fwds)
                    except:
                        h, moment, gate_top_k_idx, fwds = layer(h, h_cache[l], moment, self.key_pe, fwds = None)
                elif layer.g == 'o': # elliptical2 gate
                    try:
                        h, moment, gate_top_k_idx, moe_inp_last = layer(h, h_cache[l], moment, self.key_pe, moe_inp_last =moe_inp_last)
                    except:
                        h, moment, gate_top_k_idx, moe_inp_last = layer(h, h_cache[l], moment, self.key_pe, moe_inp_last = None)
                
                elif layer.g == 'p' or layer.g == 'c': #sparseproj gate or cosa gate
                    h, moment, gate_top_k_idx = layer(h, h_cache[l], moment, self.key_pe, gate_top_k_idx = gate_top_k_idx)
            
                else: # all other gates
                    h, moment, gate_top_k_idx = layer(h, h_cache[l], moment, self.key_pe)
                    

                #######

                # try: # set up for ellattention and standard gate
                #     h, moment, gate_top_k_idx = layer(h, h_cache[l], moment, self.key_pe, key_last = key_last, value_last = value_last)  # B x M x H
                # except:
                #     h, moment, gate_top_k_idx = layer(h,  h_cache[l], moment, self.key_pe)  # B x M x H
                #h, moment, gate_top_k_idx = layer(h, h_cache[l], moment, self.key_pe)  # B x M x H
                if l==self.layer_n:
                    #breakpoint()
                    gate_top_k_idx_out = gate_top_k_idx
            else:
                h = layer(h, [], self.key_pe)
        if self.layers[-1].s == 'e': # check if final layer was elliptical attention
            h = h[0] #  extract just attention output
        out = F.log_softmax(self.out_emb(h), dim=-1)
        if not self.finetune: # normal pretraining out
            return out, h_cache_next, gate_top_k_idx_out
        else:
            pre_logits = self.project_head(out[:, -1, :])
            return pre_logits, h_cache_next

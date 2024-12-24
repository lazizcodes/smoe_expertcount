import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
from custom_layers import FMoE
from custom_layers import FMoELinear
from custom_layers_opt import FMoEOpt


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        # print(x.shape)
        return x



class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        layerth=0,
        elliptical_gate = False,
        elliptical_gate2 = False,
        show_gate_W = False,
        mean_scale = False,
        spectral_gate =  False,
        kspectral_gate = False,
        root_invert = False,
        intra_layer =  False,
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert, d_model=d_model, moe_top_k=moe_top_k, layerth=layerth,
            elliptical_gate=elliptical_gate, elliptical_gate2=elliptical_gate2, spectral_gate=spectral_gate, kspectral_gate=kspectral_gate, show_gate_W = show_gate_W, 
            mean_scale = mean_scale, root_invert = root_invert, intra_layer=intra_layer, **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

        self.elliptical_gate = elliptical_gate
        self.elliptical_gate2 = elliptical_gate2
        self.spectral_gate = spectral_gate
        self.kspectral_gate = kspectral_gate

    def forward(self, inp: torch.Tensor, gate_top_k_idx = None, fwds = None, attn_logit = None, moe_inp_last = None, eigenvectors = None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        #breakpoint()
        if self.elliptical_gate:
            output, gate_top_k_idx, fwds = super().forward(inp, gate_top_k_idx, fwds)
            return output.reshape(original_shape), gate_top_k_idx, fwds
        elif self.elliptical_gate2:
            output, gate_top_k_idx, moe_inp_last = super().forward(inp, moe_inp_last=moe_inp_last)
            return output.reshape(original_shape), gate_top_k_idx, moe_inp_last
        if self.spectral_gate:
            output, gate_top_k_idx = super().forward(inp, attn_logit = attn_logit)
            return output.reshape(original_shape), gate_top_k_idx
        elif self.kspectral_gate:
            output, gate_top_k_idx = super().forward(inp, eigenvectors=eigenvectors)
            return output.reshape(original_shape), gate_top_k_idx
        else:
            output, gate_top_k_idx = super().forward(inp, gate_top_k_idx)
            return output.reshape(original_shape), gate_top_k_idx


class FMoETransformerMLPOpt(FMoEOpt):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            moe_top_k=moe_top_k,
            freq=freq,
            alpha=alpha,
            act_experts=act_experts,
            g_blance=g_blance,
            opt_blance=opt_blance,
            combine_gate=combine_gate,
            opt_loss=opt_loss,
            **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)

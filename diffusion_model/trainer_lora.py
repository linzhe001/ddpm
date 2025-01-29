import torch
from torch import nn
from typing import Optional, Dict
import math
from torch.nn import functional as F
from torch.nn.modules.utils import _single


class LoRALayer:
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.merged = False
        self.merge_weights = merge_weights

class LoRA1dConv(nn.Conv1d, LoRALayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, 1))
        self.scaling = self.lora_alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if not self.merged:
            def T(w):
                return w.transpose(-1, -2) if self.transpose else w
            
            lora_weight = (self.lora_B @ self.lora_A).squeeze()
            if self.padding_mode != 'zeros':
                return F.conv1d(
                    F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                    self.weight + (lora_weight.T * self.scaling),
                    self.bias,
                    self.stride,
                    _single(0),
                    self.dilation,
                    self.groups,
                )
            return F.conv1d(
                x,
                self.weight + (lora_weight.T * self.scaling),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return super().forward(x)

class LoRA3dConv(nn.Conv3d, LoRALayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
            
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, *kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, 1, 1, 1))
        self.scaling = self.lora_alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if not self.merged:
            lora_weight = (self.lora_B @ self.lora_A.view(self.lora_A.size(0), -1).unsqueeze(-1))
            lora_weight = lora_weight.view(self.weight.shape)
            return F.conv3d(
                x,
                self.weight + (lora_weight * self.scaling),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return super().forward(x)

class LoRAQKVAttention(nn.Module):
    def __init__(
        self,
        qkv: nn.Conv1d,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_qkv = qkv
        self.qkv_lora = LoRA1dConv(
            qkv.in_channels,
            qkv.out_channels,
            qkv.kernel_size[0],
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            stride=qkv.stride,
            padding=qkv.padding,
            dilation=qkv.dilation,
            groups=qkv.groups,
            bias=qkv.bias is not None,
        )
        
    def forward(self, x):
        return self.qkv_lora(x)

def add_lora_layers(model: nn.Module, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
    """
    Adds LoRA layers to the model's attention blocks and Conv3d layers
    """
    for name, module in model.named_modules():
        # Replace attention QKV layers
        if isinstance(module, AttentionBlock):
            module.qkv = LoRAQKVAttention(
                module.qkv,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            
        # Replace Conv3d layers in TimestepEmbedSequential
        if isinstance(module, TimestepEmbedSequential):
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Conv3d):
                    setattr(module, sub_name, LoRA3dConv(
                        sub_module.in_channels,
                        sub_module.out_channels,
                        sub_module.kernel_size,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        stride=sub_module.stride,
                        padding=sub_module.padding,
                        dilation=sub_module.dilation,
                        groups=sub_module.groups,
                        bias=sub_module.bias is not None,
                    ))

# Modified Trainer class initialization
class ModifiedTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(diffusion_model, dataset, **kwargs)
        # Add LoRA layers to the model
        add_lora_layers(
            self.model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        # Update optimizer to only train LoRA parameters
        self.opt = Adam(self.get_lora_parameters(), lr=kwargs.get('train_lr', 2e-6))
        
    def get_lora_parameters(self):
        """Returns only the LoRA parameters for optimization"""
        lora_params = []
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRA1dConv, LoRA3dConv)):
                lora_params.extend([module.lora_A, module.lora_B])
        return lora_params
import torch
import torch.nn as nn
import torchvision.transforms as TF


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        if torch.is_floating_point(x):
            return x.to(torch.float32)
        raise TypeError(f"Expected uint8 or floating point image tensor, got {x.dtype}")

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)

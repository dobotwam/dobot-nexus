"""Pi0-owned explicit projection casts for execution without outer AMP."""

from torch import nn
from torch.nn import functional as F


class InputDtypeLinear(nn.Linear):
    """Cast compute operands while retaining FP32 optimizer parameters."""

    def forward(self, value):
        return F.linear(value, self.weight.to(value.dtype),
                        self.bias.to(value.dtype) if self.bias is not None else None)


def configure_input_dtype_linears(core):
    """Keep Pi0 independent of other models' operator implementations."""
    for parent in list(core.modules()):
        for name, original in list(parent.named_children()):
            if type(original) is nn.Linear:
                projection = InputDtypeLinear(original.in_features, original.out_features,
                    bias=original.bias is not None, device="meta", dtype=original.weight.dtype)
                projection.weight, projection.bias = original.weight, original.bias
                projection.train(original.training)
                setattr(parent, name, projection)

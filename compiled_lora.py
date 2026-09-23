"""Compile only inference LoRA arithmetic, preserving eager precision boundaries."""
import torch
import torch.nn.functional as F


def _adapter(x, down, up, base, scale, multiplier):
    delta = F.linear(F.linear(x.to(down.dtype), down), up)
    delta = delta * scale
    delta = delta * multiplier.reshape(-1, *([1] * (base.ndim - 1)))
    return base + delta.to(base.dtype)


class CompiledLora:
    def __init__(self, network):
        # Klein has more than eight distinct adapter matrix shapes.
        # This is a finite kernel family, not repeated compilation per LoRA.
        import torch._dynamo.config
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 64)
        self.enabled = False
        self.count = 0
        self.kernel = torch.compile(_adapter, fullgraph=True, dynamic=True, options={
            'epilogue_fusion': False, 'emulate_precision_casts': True})
        for module in network.get_all_modules():
            # Keep unsupported adapters on their original path.
            if (module.__class__.__name__ != 'LoRAModule'
                    or not isinstance(module.lora_down, torch.nn.Linear)
                    or not isinstance(module.lora_up, torch.nn.Linear)
                    or module.lora_down.bias is not None
                    or module.lora_up.bias is not None
                    or getattr(module, 'lora_mid', None) is not None
                    or hasattr(module, 'scalar')
                    or isinstance(module.dropout, torch.nn.Module)):
                continue
            original = module.forward
            def forward(x, *args, _module=module, _original=original, **kwargs):
                net = _module.network_ref()
                if (not self.enabled or _module.training or not net.is_active
                        or net.is_merged_in or net.multiplier == 0
                        or x.ndim not in (2, 3) or not x.is_cuda):
                    return _original(x, *args, **kwargs)
                base = _module.org_forward(x, *args, **kwargs)
                return self.kernel(x, _module.lora_down.weight.detach(),
                    _module.lora_up.weight.detach(), base, _module._runtime_scale,
                    net.torch_multiplier)
            module.org_module[0].forward = forward
            self.count += 1

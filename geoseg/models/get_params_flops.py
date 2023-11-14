import torch
from thop import profile
from thop import clever_format
from AANet import AANet

model = AANet(num_classes=6, decoder_channels=64)
input = torch.randn([1, 3, 512, 512])
macs, params = profile(model, inputs=(input,), verbose=False)
print(f"macs = {macs / 1e9}G")
print(f"params = {params / 1e6}M")
macs, params = clever_format([macs, params], "%.3f")  # clever_format
print("Macs=", macs)
print("Params=", params)

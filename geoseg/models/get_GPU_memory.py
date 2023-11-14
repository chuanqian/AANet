import torch
from AANet import AANet

model = AANet(num_classes=6, decoder_channels=64).cuda(1)
inputs = torch.randn(4, 3, 512, 512).cuda(1)
# Get initial memory usage
torch.cuda.reset_peak_memory_stats(1)
memory_before = torch.cuda.memory_allocated(1)
# Run model forward
outputs = model(inputs)
# Get final memory usage
memory_after = torch.cuda.memory_allocated(1)
memory_usage = (memory_after - memory_before) / (1024 ** 2)  # in MB
print(memory_usage / 4)

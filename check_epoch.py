import torch
ckpt = torch.load('runs/pignn/pignn_checkpoint.pt', map_location='cpu')
print('Best Epoch:', ckpt['epoch'])
print('Best Val Loss:', ckpt['best_loss'])

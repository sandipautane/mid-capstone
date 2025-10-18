# Install if needed
#!pip install torch-lr-finder

import torch.optim as optim
from torch_lr_finder import LRFinder

# Create fresh model and optimizer
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=5e-4)  # Lower weight decay
criterion = nn.CrossEntropyLoss()


amp_config = {
    'device_type': 'cuda',
    'dtype': torch.float16,
}
grad_scaler = torch.cuda.amp.GradScaler()

lr_finder = LRFinder(
    model, optimizer, criterion, device='cuda',
    amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
)
lr_finder.range_test(train_loader, end_lr=0.1, num_iter=300, step_mode='exp')
lr_finder.plot()
lr_finder.reset()
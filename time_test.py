import sys
sys.path.append('../FLamingo/')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import time
from models import create_model_instance_custom

from FLamingo.core.utils.data_utils import ClientDataset

def measure_training_time(model, dataloader, num_epochs=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    total_forward_time = 0
    total_backward_time = 0
    total_batches = 0

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 测量前向传播时间
            start_time = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            forward_time = time.time() - start_time
            total_forward_time += forward_time

            # 测量反向传播时间
            start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - start_time
            total_backward_time += backward_time

            total_batches += 1

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Forward time: {forward_time:.4f}s, '
                      f'Backward time: {backward_time:.4f}s')

    avg_forward_time = total_forward_time / total_batches
    avg_backward_time = total_backward_time / total_batches
    
    print(f'\nAverage time per batch:')
    print(f'Forward pass: {avg_forward_time:.4f}s')
    print(f'Backward pass: {avg_backward_time:.4f}s')
    print(f'Total: {avg_forward_time + avg_backward_time:.4f}s')
    print(f'Ratio (Backward/Forward): {avg_backward_time/avg_forward_time:.2f}')

# 使用函数
model = create_model_instance_custom('alexnet','cifar10')
ds = ClientDataset('cifar10', '../datasets/cifar10_nc30_distiid_blc1', 0)
dataloader = ds.get_train_loader(32)
measure_training_time(model, dataloader)

# ++++++++++++++++++++++++++++++++
# 代码运行结果
# ++++++++++++++++++++++++++++++++
# BS=32
# Average time per batch:
# Forward pass: 0.0019s
# Backward pass: 0.0043s
# Total: 0.0061s
# Ratio (Backward/Forward): 2.29

# BS=64
# Average time per batch:
# Forward pass: 0.0017s
# Backward pass: 0.0037s
# Total: 0.0055s
# Ratio (Backward/Forward): 2.15

# BS=128
# Average time per batch:
# Forward pass: 0.0022s
# Backward pass: 0.0044s
# Total: 0.0066s
# Ratio (Backward/Forward): 1.98
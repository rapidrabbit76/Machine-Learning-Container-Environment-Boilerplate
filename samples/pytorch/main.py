from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import numpy as np


writer = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    pbar: tqdm,
):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 10 == 0:
            pbar.set_description_str(f"[{epoch}]: {loss.item(): .4f}")
    return np.mean(losses)


def main():
    dataset_dir = "./dataset"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        dataset_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(dataset_dir, train=False, transform=transform)

    dataloader_kwargs = dict(
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, **dataloader_kwargs)

    model = Net().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=0.005)

    for epoch in range(10):
        pbar = tqdm(train_loader)
        loss = train(model, optimizer, epoch, pbar=pbar)
        writer.add_scalar("loss", loss, global_step=epoch)


if __name__ == "__main__":
    main()

"""
Just a simple training script to quickly test hardware compatibility of torch and cuda.
"""

import torch
import torchvision


class CIFAR10_Net(torch.nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.in_layer = torch.nn.Conv2d(3, 8, (5, 5))
        self.hidden_conv = torch.nn.Conv2d(8, 16, (5, 5))
        self.pool = torch.nn.MaxPool2d((2, 2), 2)
        self.flat = torch.nn.Flatten()
        self.hidden_fc = torch.nn.Linear(400, 100)
        self.out_layer = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.hidden_conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.hidden_fc(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x


def train_pass(data_loader, model, loss_fn, optim):
    model.train()
    running_loss =  0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        running_loss += loss.item()

        if i % 100 == 99:
            print('Training loss: ' + str(running_loss/100))
            running_loss = 0.0
    return None


def valid_pass(data_loader, model, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    print('Validation loss: ' + str(val_loss / len(data_loader)))
    return None


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

    net = CIFAR10_Net().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

    for epoch in range(2):
        print('Epoch Number: ' + str(epoch))
        print('Entering training loop:')
        train_pass(train_loader, net, criterion, optimizer)
        print('Entering validation loop:')
        valid_pass(val_loader, net, criterion)
    print('Finished Training')

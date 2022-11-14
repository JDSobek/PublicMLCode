"""
Testing a single-path multi-scale convolutional layer.  See pytorch tutorial for better documentation of the non-novel network code.
"""


import torch


class MultiScale2D(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # suggested to use 3x3 to take advantage of optimized algorithms
                 num_one_by_one=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(MultiScale2D, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        # create tensor to make some channels equivalent to 1x1
        # this is designed so a number of channels set by num_one_by_one will have only the center element be non-zero
        # while the rest of the channels will have the entire kernel trainable
        # gradients will also be multiplied by this factor such that only the center element in the pseudo 1x1 layers change
        center_tensor = torch.zeros((num_one_by_one, in_channels, kernel_size, kernel_size))
        center_tensor[:, :, kernel_size // 2, kernel_size // 2] = 1.
        full = torch.ones((out_channels - num_one_by_one, in_channels, kernel_size, kernel_size))
        # registering the tensor as a buffer so the multiplication factor will inherit its device from the model
        self.register_buffer('mul_factor', torch.cat((center_tensor, full), dim=0))
        
        # setting initial convolutional layer weights, zeroing the non-center elements of the pseudo 1x1 layers
        data = torch.rand((out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(data)
        param = torch.nn.Parameter(torch.mul(data, self.mul_factor))
        self.conv.weight = param
        
        # register a hook to keep the gradient for the zero parts of the weights zero
        self.conv.weight.register_hook(lambda grad: torch.mul(grad, self.mul_factor))

    def forward(self, x):
        return self.conv(x)



class CIFAR10_Net(torch.nn.Module):
  def __init__(self):
    super(CIFAR10_Net, self).__init__()
    self.in_layer = MultiScale2D(3, 8, 3)
    self.hidden_conv = MultiScale2D(8, 16, 3)
    self.pool = torch.nn.MaxPool2d((2, 2), 2)
    self.flat = torch.nn.Flatten()
    self.hidden_fc = torch.nn.Linear(576, 100)
    self.out_layer = torch.nn.Linear(100, 10)
    self.relu = torch.nn.ReLU()
    
    self.count = 0
  
  def forward(self, x):   
    self.count += 1
    
    x = self.in_layer(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.hidden_conv(x)
    
    # This is just to verify that the one by one channels have stayed pseudo one by one
    if self.count % 1000 == 0:
      print('weights')
      print(self.hidden_conv.conv.weight)
    
    
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
    import torchvision
    transform = torchvision.transforms.ToTensor()

    # Set this before running
    CIFAR10_dir = r""

    train_dataset = torchvision.datasets.CIFAR10(CIFAR10_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = torchvision.datasets.CIFAR10(CIFAR10_dir, train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    
    net = CIFAR10_Net().cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)
    
    for epoch in range(5):
        print('Epoch Number: ' + str(epoch))
        print('Entering training loop:')
        train_pass(train_loader, net, criterion, optimizer)
        print('Entering validation loop:')
        valid_pass(val_loader, net, criterion)
        print('Finished Training')    

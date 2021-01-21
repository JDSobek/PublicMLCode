import os
from random import randint
import math
import torch
import torchvision
from torch.utils import data
from torch.nn.modules.loss import _Loss
from PIL import Image


def cover_patch(image_tensor: torch.Tensor) -> torch.Tensor:
    """Creates a 3-D tensor that is the input 3-D tensor with random rectangular patches zeroed out."""
    channels, image_x_len, image_y_len =  image_tensor.size()
    num_covers = randint(1,5)
    covered_image = torch.clone(image_tensor)
    for i in range(num_covers):
        patch_x_len = randint(math.floor(image_x_len/5), math.floor((image_x_len/3)))
        patch_y_len = randint(math.floor(image_y_len/5), math.floor((image_y_len/3)))
        patch_x_start = randint(0, image_x_len-patch_x_len-1)
        patch_y_start = randint(0, image_y_len-patch_y_len-1)
        covered_image[:, patch_x_start:patch_x_start+patch_x_len, patch_y_start:patch_y_start+patch_y_len] = torch.zeros(size=[channels, patch_x_len, patch_y_len])
    return covered_image


def gramMatrix(activations: torch.Tensor) -> torch.Tensor:
    """Calculates the Gram matrix for a 4-D tensor (batch, channels, height, width)
     using the efficient method suggested in https://arxiv.org/pdf/1603.08155.pdf"""
    batch, channels, _, _ = activations.size()
    # take the mean here to normalize the Gram matrix for later steps
    norm_factor = 1./torch.numel(activations)
    # calculate the Gram matrix
    flattened_activations = torch.reshape(activations, (batch*channels, -1,))
    gram_matrix = norm_factor*torch.matmul(flattened_activations, flattened_activations.t())
    return gram_matrix


def layerStyleLoss(pred_activations: torch.Tensor, target_activations: torch.Tensor):
    """Calculates the style loss between two sets of activations.
    Activations must be 4-D tensors in (batch, channels, height, width) format."""
    if pred_activations.size()[0] != target_activations.size()[0] or pred_activations.size()[1] != target_activations.size()[1]:
        raise AttributeError("Both sets of activations must have the same number of channels and equal batch size.")
    # use reduction='sum' because the Gram matrix was already divided by batch*channels*height*width
    # using reduction='mean' will divide result by an additional (batch*channel)^2
    return torch.nn.functional.mse_loss(gramMatrix(pred_activations), gramMatrix(target_activations), reduction='sum')


class CustomDataset(data.Dataset):
    # define the dataset used for the style transfer model
    def __init__(self, data_dir):
        #tensor_dir = data_dir
        content_dir = data_dir
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(10),
                                                        torchvision.transforms.Resize(256),
                                                        torchvision.transforms.RandomCrop(256),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                        ])

        # fill the dictionaries that store the data locations
        self.content_dict = {}
        for dirpath, subdirs, files in os.walk(content_dir):
            for file in files:
                key = file
                self.content_dict[key] = os.path.join(dirpath, file)
        self.content_keys = list(self.content_dict.keys())
        self.content_len = len(self.content_keys)

        #self.style_dict = {}
        # for dirpath, subdirs, files in os.walk(style_dir):
        #     for file in files:
        #         key = file
        #         self.style_dict[key] = os.path.join(dirpath, file)
        #self.style_keys = list(self.style_dict.keys())
        #self.style_len = len(self.style_keys)

    def __getitem__(self, index):
        # get an item from the dataset
        content_img = Image.open(self.content_dict[self.content_keys[index]]).convert('RGB')
        content_img = self.transforms(content_img).cuda()

        #style_index = randint(0, self.style_len-1)
        #style_img = Image.open(self.style_dict[self.style_keys[style_index]]).convert('RGB')
        #style_img = self.transforms(style_img).cuda()

        x_data = cover_patch(content_img)
        y_data = content_img #, style_img

        # some code to save out examples of what you're feeding the model
        # convert_back = torchvision.transforms.ToPILImage()
        # inverse_norm = torchvision.transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #                                                 std=[1. / 0.229, 1. / 0.224, 1. / 0.225])
        # test_output = convert_back(inverse_norm(covered_image).cpu())
        # file_name = r"C:\Users\Joe\Pictures\style_transfer\modeltest_"+ str(index) +".jpg"
        # test_output.save(file_name)

        return x_data, y_data

    def __len__(self):
        # return the length of the dataset
        return self.content_len


class UNet_down_block(torch.nn.Module):
    # define the down blocks for the U-Net
    def __init__(self, input_channel, output_channel, down_size):
        # initialize the layers used for the down block
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        # define what happens in this block during the forward pass for the U-Net
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_up_block(torch.nn.Module):
    # define the up block for the U-Net
    def __init__(self, prev_channel, input_channel, output_channel):
        # initialize the layers used for the up block
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        # define what happens in this block during the forward pass for the U-Net
        # print(x.size())
        # print(prev_feature_map.size())
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class StyleNet(torch.nn.Module):
    # define the U-Net architecture
    def __init__(self):
        # initialize the layers used to define the architecture
        super(StyleNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 3, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # define what happens during the forward pass for this U-Net
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class PerceptualLoss(_Loss):
    def __init__(self, style_path) -> None:
        super(PerceptualLoss, self).__init__()
        self.style_path = style_path
        self.ref_model = torchvision.models.vgg19(pretrained=True).cuda().features #torchvision.models.resnet18(pretrained=True).cuda()
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                     ])

        # this code won't work with more than one style target but doing it this way lets me avoid hardcoding a filename
        # for dirpath, subdirs, files in os.walk(self.style_path):
        #     for file in files:
        #         style_img = Image.open(dirpath+file).convert('RGB')
        #         style_img = self.transforms(style_img).cuda()
        #         # doesn't go out of scope because python
        #         self.style_target = torch.unsqueeze(style_img, dim=0)

        self.style_dict = {}
        for dirpath, subdirs, files in os.walk(self.style_path):
            for file in files:
                key = file
                self.style_dict[key] = os.path.join(dirpath, file)
        self.style_keys = list(self.style_dict.keys())
        self.style_len = len(self.style_keys)

        # layers at which we will calculate loss
        # layers just before max pooling layers: 34, 25, 16, 7, 2
        # this dictionary format will probably only work for vgg, would like to make it work for resnet too
        # entries formatted as below
        # layer: (layer content weight, layer style weight)
        self.layers = {
            7: (1, 1),
            16: (1, 1),
            25: (1, 1),
            34: (1, 1)
        }

    def forward(self, preds, target):
        batch_loss = 0.
        mse_reduce = 'mean' # reduction method for the content loss calculated below

        # note class variables shouldn't normally be declared outside __init__ but i want to maximize ability
        # to try new methodologies while I test and this is the easiest way to allow me to switch
        # between using single or multiple style targets
        style_index = randint(0, self.style_len-1)
        style_img = Image.open(self.style_dict[self.style_keys[style_index]]).convert('RGB')
        self.style_target = torch.unsqueeze(self.transforms(style_img).cuda(), dim=0)

        # repeating the style target is necessary for the Gram matrices to be calculated
        # without flattening along the batch axis
        batch_size, channel_num, width, height = preds.size()
        style_target = self.style_target.repeat(batch_size, 1, 1, 1)

        # calculate the losses at the output of the UNet
        output_content_loss = torch.nn.functional.mse_loss(preds, target, reduction=mse_reduce)
        output_style_loss = layerStyleLoss(preds, style_target)
        # print("content loss at output:")
        # print(output_content_loss)
        # print("style loss at output:")
        # print(output_style_loss)
        batch_loss += output_content_loss + output_style_loss

        # calculate the losses for the output from the reference model's layers
        for layer in self.layers.keys():
            ref_target = self.ref_model[:layer+1](target)
            ref_preds = self.ref_model[:layer+1](preds)
            ref_style = self.ref_model[:layer+1](style_target)
            batch_size, channel_num, width, height = ref_preds.size()
            ref_content_loss = torch.nn.functional.mse_loss(ref_preds, ref_target, reduction=mse_reduce)
            ref_style_loss = layerStyleLoss(ref_preds, ref_style)
            # print("style loss at layer" + str(layer))
            # print(ref_style_loss)
            # print("content loss at layer " + str(layer))
            # print(ref_content_loss)
            # multiplying by respective weights
            batch_loss += self.layers[layer][0]*ref_style_loss + self.layers[layer][0]*ref_content_loss

        return batch_loss


if __name__ == '__main__':
    num_epochs = 1000
    train_bs = 2
    val_bs = 2
    # directories where the images I'm using are stored
    # parent_path = r"C:\Users\Joe\Pictures\style_transfer\\"
    content_path = r"C:\Users\Joe\Pictures\style_transfer\content2\\"
    style_path = r"C:\Users\Joe\Pictures\style_transfer\style2\\"

    test_path = content_path + r"val\\"
    train_path = content_path + r"train\\"

    # create model
    net = StyleNet().cuda()
    #print(net)

    # create the train and validation datasets
    trainset = CustomDataset(train_path)
    testset = CustomDataset(test_path)
    # create dataloaders from the datasets
    trainloader = data.DataLoader(trainset, batch_size=train_bs, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=val_bs, shuffle=True)
    print('Finished Dataset Creation')

    # define the model's loss, optimizer, and scheduler
    criterion = PerceptualLoss(style_path=style_path)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.3)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.03, epochs=num_epochs, steps_per_epoch=len(trainloader))

    print('Beginning Training')
    # define the training loop and train the model
    loss_accum_size = 1 # print training loss after every batch
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            # print training loss
            if i % loss_accum_size == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/loss_accum_size))
                running_loss = 0.0

        val_loss = 0.0
        val_counter = 0
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_counter += 1
        # print validation loss
        print('[%d, %5d] val loss: %.3f' % (epoch + 1, i + 1, val_loss / val_counter))
        # one-cycle scheduler steps every batch not every epoch
        #scheduler.step()

    print('Finished Training')

    # save trained model
    modelfilesavepath = r"C:\Users\Joe\Documents\Py3Programs\fastaitests\styletransfernet.pth"
    torch.save(net.state_dict(), modelfilesavepath)
    print('Finished Saving')

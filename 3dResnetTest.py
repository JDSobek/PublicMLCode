import torch
import math
from ResNet3DUNet import ResNet3DUNet
from TorchResnet3dconvs import resnet34
from dataset import *
from losses import DiceLossSoftmax
from timeit import default_timer as timer
from utils import *


def CE_train_loop(data_loader, model, loss_fn, grad_optimizer, epoch_num, grad_scaler=None, augment = False):
    """Define the training loop.  Note training in function scope like this helps avoid CUDA OOM errors"""
    # set model in train mode
    model.train()
    data_len = len(data_loader.dataset)
    data_batches = math.ceil(data_len / data_loader.batch_size)
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        #grad_optimizer.zero_grad()
        # more efficient way to zero gradient
        for param in model.parameters():
            param.grad = None

        # using automated mixed precision training
        if grad_scaler is not None:
            inputs, masks = data
            inputs = inputs.cuda()
            masks = masks.long().cuda()
            with torch.no_grad():
                inputs, _ = input_prep(inputs, masks, data_loader.dataset.mask_labels)
                if augment:
                    inputs, masks = augment_fn(inputs, masks)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, masks)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(grad_optimizer)
            grad_scaler.update()
        # not using automated mixed precision training
        else:
            inputs, masks = data
            inputs = inputs.cuda()
            masks = masks.cuda()
            with torch.no_grad():
                inputs, _ = input_prep(inputs, masks, data_loader.dataset.mask_labels)
            if augment:
                with torch.no_grad():
                    inputs, masks = augment_fn(inputs, masks)
            outputs = model(inputs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            grad_optimizer.step()

        running_loss += loss.item()
        # explicitly clearing memory to prevent memory leaks
        del inputs, masks, outputs
        if i % 10 == 9:
            print('[Epoch %d, Batch %5d/' % (epoch_num + 1, i + 1) + str(data_batches) + '] training loss: %.3f' % (running_loss / 10))
            running_loss = 0.0
        # print('[Epoch %d, Batch %5d/' % (epoch + 1, i + 1) + str(data_batches) + '] training loss: %.3f' % (running_loss))
        # running_loss = 0.0
    return None


def train_loop(data_loader, model, loss_fn, grad_optimizer, epoch_num, grad_scaler=None, augment = False):
    """Define the training loop.  Note training in function scope like this helps avoid CUDA OOM errors"""
    # set model in train mode
    model.train()
    data_len = len(data_loader.dataset)
    data_batches = math.ceil(data_len / data_loader.batch_size)
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        #grad_optimizer.zero_grad()
        # more efficient way to zero gradient
        for param in model.parameters():
            param.grad = None

        # using automated mixed precision training
        if grad_scaler is not None:
            inputs, masks = data
            inputs = inputs.cuda()
            masks = masks.cuda()
            with torch.no_grad():
                inputs, masks = input_prep(inputs, masks, data_loader.dataset.mask_labels)
                if augment:
                    inputs, masks = augment_fn(inputs, masks)
            with torch.cuda.amp.autocast():
                outputs = torch.nn.functional.softmax(model(inputs), dim=1)
                loss = loss_fn(outputs, masks)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(grad_optimizer)
            grad_scaler.update()
        # not using automated mixed precision training
        else:
            inputs, masks = data
            inputs = inputs.cuda()
            masks = masks.cuda()
            with torch.no_grad():
                inputs, masks = input_prep(inputs, masks, data_loader.dataset.mask_labels)
            if augment:
                with torch.no_grad():
                    inputs, masks = augment_fn(inputs, masks)
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            loss = loss_fn(outputs, masks)
            loss.backward()
            grad_optimizer.step()

        running_loss += loss.item()
        # explicitly clearing memory to prevent memory leaks
        del inputs, masks, outputs
        if i % 10 == 9:
            print('[Epoch %d, Batch %5d/' % (epoch_num + 1, i + 1) + str(data_batches) + '] training loss: %.3f' % (running_loss / 10))
            running_loss = 0.0
        # print('[Epoch %d, Batch %5d/' % (epoch + 1, i + 1) + str(data_batches) + '] training loss: %.3f' % (running_loss))
        # running_loss = 0.0
    return None


def val_loop(data_loader, model, loss_fn, epoch_num):
    """Define a validation loop.  Note training in function scope like this helps avoid CUDA OOM errors"""
    # set model in eval mode
    model.eval()
    data_len = len(data_loader.dataset)
    data_batches = math.ceil(data_len / data_loader.batch_size)
    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, masks = data
            inputs = inputs.cuda()
            masks = masks.cuda()
            inputs, masks = input_prep(inputs, masks, data_loader.dataset.mask_labels)
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            # explicitly clearing memory to prevent memory leaks
            del inputs, masks, outputs

    print('[Epoch %d] ' % (epoch_num + 1) + 'val loss: %.3f' % (val_loss / data_batches))
    return None


if __name__ == '__main__':
    # limit what GPUs this program runs on
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # set directories
    datadir = "/Data/test_tensors/"
    modelfilesavepath = r"/Models/3DResnetUNet.pth"

    # set hyperparameters
    first_pretrain_epochs = 2
    second_pretrain_epochs = 2
    train_epochs = 10
    bs = 18
    adam_lr = 0.0003
    max_lr = 0.03
    save_model = True
    # num_workers > 0 requires spawn start method for CUDA processes
    # torch.multiprocessing.set_start_method('spawn')
    # in general this works easier if you push the entire batch to GPU at once instead of doing it in the
    # dataset
    workers = 4
    # memory pinning only works if the data is pushed to GPU in the training loop
    memory_pinning = True

    print("Instantiating model")
    resnetmodel = resnet34(pretrained=True, is_encoder=True)
    unetmodel = ResNet3DUNet(resnetmodel, input_depth=5, num_classes=2).cuda()

    count_parameters(unetmodel)

    print("Setting up Dataloaders and optimizer")
    trainset = MedDataset(datadir, test=False)
    testset = MedDataset(datadir, test=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=memory_pinning)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=memory_pinning)


    first_pretrain_criterion = torch.nn.functional.cross_entropy
    second_pretrain_criterion = DiceLossSoftmax()
    criterion = DiceLossSoftmax()
    optimizer = torch.optim.Adam(unetmodel.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=train_epochs, steps_per_epoch=len(trainloader))
    scaler = torch.cuda.amp.GradScaler()


    print("Annealing the randomly initialized layers")
    # first pre-training (Cross Entropy)
    print("Cross Entropy annealing")
    for epoch in range(first_pretrain_epochs):
        CE_train_loop(trainloader, unetmodel, first_pretrain_criterion, optimizer, epoch, scaler, augment=True)
        val_loop(testloader, unetmodel, criterion, epoch)

    # second pre-training (Dice)
    print("Dice Loss annealing")
    for epoch in range(second_pretrain_epochs):
        train_loop(trainloader, unetmodel, second_pretrain_criterion, optimizer, epoch,scaler, augment=True)
        val_loop(testloader, unetmodel, criterion, epoch)

    # unfreeze all layers
    for param in unetmodel.parameters():
        param.requires_grad = True

    # training loop
    print("Beginning full model training")
    for epoch in range(train_epochs):
        begin = timer()
        train_loop(trainloader, unetmodel, criterion, optimizer, epoch, scaler, augment=True)
        val_loop(testloader, unetmodel, criterion, epoch)
        end = timer()
        print("Epoch duration: " + str(math.floor(end-begin)) + " seconds.")
        scheduler.step()
    print("Finished Training")

    if save_model:
        print("Saving Model")
        torch.save(unetmodel.state_dict(), modelfilesavepath)
        print("Finished Saving")

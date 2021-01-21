import torch


@torch.jit.script
def func_dice_loss_softmax(preds: torch.Tensor, targ: torch.Tensor):
    """softmax version of soft Dice loss, uses pseudo label smoothing on the predictions
    note that this is different from the fast.ai version because my predictions and targets
    have a different shape in this torch code than in my fast.ai code"""
    eps = 1.e-5 # prevents div by 0
    pred = preds.cuda()
    targ = targ.cuda()

    # apply label smoothing
    tens_0 = torch.zeros(pred.size()).cuda()
    tens_1 = torch.ones(pred.size()).cuda()
    pred = torch.where(pred<0.1, tens_0, pred)
    pred = torch.where(pred>0.9, tens_1, pred)

    # background is index 2 channel 0, worth trying training both with and without it
    inter = torch.sum(pred[:, 1:,...] * targ[:, 1:,...])
    union = torch.sum(pred[:, 1:,...] + targ[:, 1:,...])
    loss = 1. - (2. * inter + eps) / (union + eps)
    return loss


class DiceLossSoftmax(torch.nn.modules.loss._Loss):
    """Instantiate the multi-class label smoothed soft Dice loss as a torch loss module"""
    def __init__(self) -> None:
        super(DiceLossSoftmax, self).__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return func_dice_loss_softmax(preds, target)


@torch.jit.script
def func_dice_loss_sigmoid(preds: torch.Tensor, targ: torch.Tensor):
    """may not have been fixed to work with 3d input data"""
    # torch losses, like those subclassed by fastai losses, call out to basic functions like this
    eps = 1.e-5
    pred = torch.squeeze(torch.sigmoid(preds), dim=1) # need dimensions to match in flattened vectors
    inter = torch.sum(pred * targ)
    union = torch.sum(pred + targ)
    loss = 1. - (2. * inter + eps) / (union + eps)
    return loss


class DiceLossSigmoid(torch.nn.modules.loss._Loss):
    # fastai losses subclass base classes from torch
    def __init__(self) -> None:
        super(DiceLossSigmoid, self).__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return func_dice_loss_sigmoid(preds, target)


@torch.jit.script
def gramMatrix(activations: torch.Tensor) -> torch.Tensor:
    """Calculates the Gram matrix for a 3-D tensor (Channels, height, width)
     using the efficient method suggested in https://arxiv.org/pdf/1603.08155.pdf"""
    channels, height, width = activations.size()
    norm_factor = 1. / (channels * height * width)
    semi_flattened_activations = torch.reshape(activations, (channels, -1,))
    transposed_semi_flattened_activations = torch.transpose(semi_flattened_activations, 0, 1)
    gram_matrix = norm_factor * torch.matmul(semi_flattened_activations, transposed_semi_flattened_activations)
    return gram_matrix


def layerStyleLoss(pred_activations: torch.Tensor, target_activations: torch.Tensor):
    """Calculates the style loss between two sets of activations.
    Activations must be 3-D tensors in (Channels, height, width) format."""
    if pred_activations.size()[0] != target_activations.size()[0]:
        raise AttributeError("Both sets of activations must have the same number of channels.")
    pred_gram = gramMatrix(pred_activations)
    target_gram = gramMatrix(target_activations)
    gram_diff = torch.sub(pred_gram, target_gram)
    squared_frob_norm = torch.sum(torch.square(gram_diff))
    return squared_frob_norm


def foreground_acc_sigmoid(inp, targ, bkg_idx=0):
    """Computes non-background accuracy for single class segmentation.
    Mask must have all classes on same plane."""
    # remove singleton dimension at index 1 for the comparison
    inp = torch.squeeze(inp, dim=1)
    mask = (targ != bkg_idx)
    return ((inp>0.5)[mask] == targ[mask]).float().mean()

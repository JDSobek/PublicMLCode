import torch


def getCenter(input_tensor: torch.Tensor) -> torch.Tensor:
    """grabs the center slice of a multi-slice 5-d array
    only provides a true center if there is an odd number of slices
    note data is arranged as [batch, channel, depth, height, width]"""
    center = input_tensor.size()[2]//2
    return torch.unsqueeze(input_tensor[:,:,center,:,:].clone(), dim=2)


def count_parameters(model):
    """A function that counts the total number of parameters in a model"""
    num_param =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num Parameters: " + str(num_param))
    return None

if __name__ == '__main__':
    pass

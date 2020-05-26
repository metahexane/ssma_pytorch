import torch

def calculate_loss(x, labels, eps=10**-10):
    log_loss = torch.log(torch.softmax(x, dim=1) + eps)
    log_loss = labels * log_loss
    return -torch.sum(log_loss)
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def mse_loss(alpha, target, num_classes, kl_scale=1.0):
    """
    EDL Mean Squared Error loss with KL regularization.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    one_hot = F.one_hot(target, num_classes).float()
    mse = torch.sum((one_hot - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
    loss = mse + var
    kl = kl_divergence(alpha, one_hot)
    return torch.mean(loss + kl_scale * kl)


def nll_loss(alpha, target, num_classes, kl_scale=1.0):
    """
    EDL Negative Log-Likelihood Loss.
    """
    y = F.one_hot(target, num_classes).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum(y * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
    kl = kl_divergence(alpha, y)
    return torch.mean(-loglikelihood + kl_scale * kl)


def digamma_loss(alpha, target, num_classes, kl_scale=1.0):
    """
    EDL Digamma Loss.
    """
    one_hot = F.one_hot(target, num_classes).float()
    alpha0 = torch.sum(alpha, dim=1, keepdim=True)
    loss = torch.sum(one_hot * (torch.digamma(alpha0) - torch.digamma(alpha)), dim=1, keepdim=True)
    kl = kl_divergence(alpha, one_hot)
    return torch.mean(loss + kl_scale * kl)


def kl_divergence(alpha, y):
    """
    KL divergence between Dirichlet(alpha) and uniform Dirichlet.
    """
    alpha_tilde = y + (1 - y) * alpha
    K = alpha.shape[1]
    
    # KL Divergence between Dir(alpha_tilde) || Dir(uniform)
    uniform = torch.ones_like(alpha)
    kl_div = (
        torch.lgamma(torch.sum(alpha_tilde, dim=1)) - torch.sum(torch.lgamma(alpha_tilde), dim=1)
        - torch.lgamma(torch.sum(uniform, dim=1)) + torch.sum(torch.lgamma(uniform), dim=1)
        + torch.sum((alpha_tilde - uniform) * (torch.digamma(alpha_tilde) - torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))), dim=1)
    )
    return kl_div


def get_loss_function(loss_type):
    """
    Returns the appropriate loss function depending on the loss_type string.
    """
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'nll':
        return nll_loss
    elif loss_type == 'digamma':
        return digamma_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

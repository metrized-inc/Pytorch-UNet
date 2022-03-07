import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_cross_entropy_loss(preds, edges):
    beta = 0.75
    """Calculate sum of weighted cross entropy loss."""
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    edges_temp = edges[:, None, :, :]
    mask = (edges_temp > 0.5).float()
    b, c, h, w = mask.shape
    # b, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1,2,3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos  # Shape: [b,].
    weight = torch.zeros_like(mask)  # B x C x H x W


    wp = ((1 - beta) * num_neg / (num_pos + num_neg)).view(b, 1, 1, 1)
    wn = (beta * num_pos / (num_pos + num_neg)).view(b, 1, 1, 1)

    # print('wp: {}'.format(wp))
    # print('wn: {}'.format(wn))

    # Express the mask as a weighted sum
    weight = torch.mul(mask, wp) + torch.mul(1 - mask, wn)

    # print('\nweights: {}'.format(weight.shape))
    # print('edges: {}'.format(edges.shape))
    # print('preds: {}'.format(preds.shape))
    

    # Calculate loss using logits function (autocasting into 16 bit precision available)
    logp = F.log_softmax(preds, dim=1)
    logp = logp.gather(1, edges.view(b, 1, h, w))
    weighted_logp = (logp * weight).view(b, -1)
    weighted_loss = weighted_logp.sum(1) / weight.view(b, -1).sum(1)
    weighted_loss = -1*weighted_loss.mean()

    # loss_func = nn.CrossEntropyLoss(weight=weight)
    # weighted_loss = loss_func(preds, edges)


    # preds = torch.sigmoid(preds)
    # losses = torch.nn.functional.binary_cross_entropy(
    #     preds.float(), edges.float(), weight=weight, reduction="None"
    # )
    # loss = torch.sum(losses) / b  # Average

    return weighted_loss
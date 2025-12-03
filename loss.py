import torch
import torch.nn.functional as F


def sobel_filters(device, channels=1):
    # 3x3 Sobel kernels
    kernel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], device=device)
    kernel_y = torch.tensor([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]], device=device)

    # shape [1,1,3,3] then repeat for groups
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    return kernel_x, kernel_y

def sobel_gradients(x):
    """
    x: [B, C, H, W], assume already on correct device
    returns gradient magnitude map [B, C, H, W]
    """
    b, c, h, w = x.shape
    device = x.device
    kx, ky = sobel_filters(device, channels=c)

    # depthwise conv: groups=c
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)

    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return grad_mag


def gradient_loss(x_pred, x_true, reduction="mean"):
    """
    L1 loss between Sobel gradient magnitudes.
    """
    g_pred = sobel_gradients(x_pred)
    g_true = sobel_gradients(x_true)
    if reduction == "mean":
        return torch.mean(torch.abs(g_pred - g_true))
    elif reduction == "sum":
        return torch.sum(torch.abs(g_pred - g_true))
    else:
        # no reduction
        return torch.abs(g_pred - g_true)

class SobelReconstructionLoss(torch.nn.Module):
    def __init__(self, pix_weight=1.0, grad_weight=0.1, use_l1=False):
        super().__init__()
        self.pix_weight = pix_weight
        self.grad_weight = grad_weight
        if use_l1:
            self.pix_loss_fn = torch.nn.L1Loss()
        else:
            self.pix_loss_fn = torch.nn.MSELoss()

    def forward(self, pred, target):
        # pixel-level loss
        lpix = self.pix_loss_fn(pred, target)

        # gradient-level loss
        lgrad = gradient_loss(pred, target, reduction="mean")

        loss = self.pix_weight * lpix + self.grad_weight * lgrad
        return loss, lpix, lgrad

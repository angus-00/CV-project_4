"""Evaluation metrics for 3D reconstruction"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import lpips


def compute_psnr(img1: Union[torch.Tensor, np.ndarray],
                 img2: Union[torch.Tensor, np.ndarray],
                 max_val: float = 1.0) -> float:
    """Compute PSNR between two images

    Args:
        img1: First image (H, W, C) or (C, H, W)
        img2: Second image (H, W, C) or (C, H, W)
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR value in dB
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1: Union[torch.Tensor, np.ndarray],
                 img2: Union[torch.Tensor, np.ndarray],
                 window_size: int = 11) -> float:
    """Compute SSIM between two images

    Args:
        img1: First image (H, W, C) or (C, H, W)
        img2: Second image (H, W, C) or (C, H, W)
        window_size: Size of the Gaussian window

    Returns:
        SSIM value between 0 and 1
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    # Convert (H, W, C) to (C, H, W) if needed
    if img1.dim() == 3 and img1.shape[-1] == 3:
        img1 = img1.permute(2, 0, 1)
    if img2.dim() == 3 and img2.shape[-1] == 3:
        img2 = img2.permute(2, 0, 1)

    # Add batch dimension if needed
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


_lpips_model = None

def compute_lpips(img1: Union[torch.Tensor, np.ndarray],
                  img2: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute LPIPS between two images

    Args:
        img1: First image (H, W, C) or (C, H, W), range [0, 1]
        img2: Second image (H, W, C) or (C, H, W), range [0, 1]

    Returns:
        LPIPS value (lower is better)
    """
    global _lpips_model

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    # Get device from input tensors
    device = img1.device if img1.is_cuda else 'cpu'

    # Initialize LPIPS model on the correct device
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
    elif next(_lpips_model.parameters()).device != img1.device:
        _lpips_model = _lpips_model.to(device)

    # Convert (H, W, C) to (C, H, W) if needed
    if img1.dim() == 3 and img1.shape[-1] == 3:
        img1 = img1.permute(2, 0, 1)
    if img2.dim() == 3 and img2.shape[-1] == 3:
        img2 = img2.permute(2, 0, 1)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # Normalize to [-1, 1] for LPIPS
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    with torch.no_grad():
        dist = _lpips_model(img1, img2)
    return dist.item()

"""3D Gaussian Splatting core module"""

# Avoid circular imports by not importing in __init__.py
# Import directly from submodules instead:
# from gaussian_splatting.model import GaussianModel
# from gaussian_splatting.trainer import GaussianTrainer
# from gaussian_splatting.renderer import GaussianRenderer

__all__ = [
    'GaussianModel',
    'GaussianTrainer',
    'GaussianRenderer',
]

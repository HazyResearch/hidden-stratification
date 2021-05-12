"""Rotation from kornia."""
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "affine",
    "rotate",
    "Rotate",
]

pi = torch.tensor(3.14159265358979323846)


def convert_affinematrix_to_homography(A: torch.Tensor) -> torch.Tensor:
    r"""Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].
    Examples::
        >>> input = torch.rand(2, 2, 3)  # Bx2x3
        >>> output = kornia.convert_affinematrix_to_homography(input)  # Bx3x3
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}".format(A.shape))
    H: torch.Tensor = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def normal_transform_pixel(height: int, width: int) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height (int): image height.
        width (int): image width.
    Returns:
        Tensor: normalized transform.
    Shape:
        Output: :math:`(1, 3, 3)`
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])  # 3x3

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    tr_mat = tr_mat.unsqueeze(0)  # 1x3x3
    return tr_mat


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destiantion to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_src (tuple): size of the destination image (height, width).
    Returns:
        Tensor: the normalized homography.
    Shape:
        Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(dst_pix_trans_src_pix):
        raise TypeError(
            "Input dst_pix_trans_src_pix type is not a torch.Tensor. Got {}".format(
                type(dst_pix_trans_src_pix)
            )
        )

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    # the devices and types
    device: torch.device = dst_pix_trans_src_pix.device
    dtype: torch.dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(device, dtype)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(device, dtype)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    flags: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Applies an affine transformation to a tensor.
    The function warp_affine transforms the source tensor using
    the specified matrix:
    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )
    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (torch.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details
    Returns:
        torch.Tensor: the warped tensor.
    Shape:
        - Output: :math:`(B, C, H, W)`
    .. note::
       See a working example `here <https://kornia.readthedocs.io/en/latest/
       tutorials/warp_affine.html>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}".format(type(src)))

    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))
    B, C, H, W = src.size()
    dsize_src = (H, W)
    out_size = dsize
    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, dsize_src, out_size)
    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, out_size[0], out_size[1]],
        align_corners=align_corners,
    )
    return F.grid_sample(
        src, grid, align_corners=align_corners, mode=flags, padding_mode=padding_mode
    )


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians.
    Args:
        tensor (torch.Tensor): Tensor of arbitrary shape.
    Returns:
        torch.Tensor: tensor with same shape as input.
    Examples::
        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = kornia.deg2rad(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    r"""Create a rotation matrix out of angles in degrees.
    Args:
        angle: (torch.Tensor): tensor of angles in degrees, any shape.
    Returns:
        torch.Tensor: tensor of *x2x2 rotation matrices.
    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(
    center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    r"""Calculates an affine matrix of 2D rotation.
    The function calculates the following matrix:
    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}
    where
    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})
    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.
    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.
    Returns:
        Tensor: the affine matrix of 2D rotation.
    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`
    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> M = kornia.get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}".format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}".format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}".format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}".format(angle.shape))
    if not len(scale.shape) == 1:
        raise ValueError("Input scale must be a B tensor. Got {}".format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Got center {}, angle {} and scale {}".format(
                center.shape, angle.shape, scale.shape
            )
        )
    # convert angle and apply scale
    scaled_rotation: torch.Tensor = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha: torch.Tensor = scaled_rotation[:, 0, 0]
    beta: torch.Tensor = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]

    # create output tensor
    batch_size: int = center.shape[0]
    one = torch.tensor(1.0).to(center.device)
    M: torch.Tensor = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y
    return M


# utilities to compute affine matrices


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane."""
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y], device=tensor.device, dtype=tensor.dtype
    )
    return center


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones_like(angle)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166


def affine(
    tensor: torch.Tensor, matrix: torch.Tensor, mode: str = "bilinear", align_corners: bool = False
) -> torch.Tensor:
    r"""Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.
        mode (str): 'bilinear' | 'nearest'
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(
        tensor, matrix, (height, width), mode, align_corners=align_corners
    )

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185


def rotate(
    tensor: torch.Tensor,
    angle: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}".format(type(angle)))
    if center is not None and not torch.is_tensor(angle):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))
    if len(tensor.shape) not in (3, 4):
        raise ValueError(
            "Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape)
        )

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, align_corners)


class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.

    Args:
        angle (torch.Tensor): The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    Returns:
        torch.Tensor: The rotated tensor.
    """

    def __init__(
        self,
        angle: torch.Tensor,
        center: Union[None, torch.Tensor] = None,
        align_corners: bool = False,
    ) -> None:
        super(Rotate, self).__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rotate(input, self.angle, self.center, align_corners=self.align_corners)

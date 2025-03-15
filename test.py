import torch
from deformable_attention import DeformableAttention

from deformable_gqa import DeformableAttention2D


def test_correctness():
    torch.manual_seed(42)
    attn_ref = DeformableAttention(
        dim=512,  # feature dimensions
        dim_head=64,  # dimension per head
        heads=8,  # attention heads
        dropout=0.0,  # dropout
        downsample_factor=4,  # downsample factor (r in paper)
        offset_scale=4,  # scale of offset, maximum offset
        offset_groups=None,  # number of offset groups, should be multiple of heads
        offset_kernel_size=6,  # offset kernel size
    )

    torch.manual_seed(42)
    attn = DeformableAttention2D(
        dim=512,  # feature dimensions
        dim_head=64,  # dimension per head
        heads=8,  # attention heads
        num_groups=8,
        dropout=0.0,  # dropout
        downsample_factor=4,  # downsample factor (r in paper)
        offset_scale=4,  # scale of offset, maximum offset
        offset_groups=None,  # number of offset groups, should be multiple of heads
        offset_kernel_size=6,  # offset kernel size
        device="cpu",
    )
    x = torch.randn(1, 512, 64, 64)
    assert torch.allclose(attn(x), attn_ref(x))

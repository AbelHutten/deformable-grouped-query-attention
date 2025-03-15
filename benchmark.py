import matplotlib.pyplot as plt
import torch

from deformable_gqa import DeformableAttention2D

batch_size = 4096
embedding_dimension = 512
width = 4
height = 4
x = torch.randn(batch_size, embedding_dimension, width, height, device="cuda")

n_warmups = 60
n_measure = 30

times = []
groups = [1, 4, 8, 16, 32]
for g in groups:
    print(g)
    attn = DeformableAttention2D(
        dim=embedding_dimension,  # feature dimensions
        dim_head=64,  # dimension per head
        heads=32,  # attention heads
        num_groups=g,
        dropout=0.0,  # dropout
        downsample_factor=4,  # downsample factor (r in paper)
        offset_scale=4,  # scale of offset, maximum offset
        offset_groups=None,  # number of offset groups, should be multiple of heads
        offset_kernel_size=6,  # offset kernel size
        device="cuda",
    )

    for _ in range(n_warmups):
        attn(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_measure):
        attn(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / n_measure)


deformable_mha_time = times[-1]
deformable_mqa_time = times[0]

tick_labels = [1 << n for n in range(6)]
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.axhline(deformable_mha_time, color="r", linestyle="--", label="Deformable MHA")
plt.semilogx(groups, times, "o-", subs=groups, base=2, label="Deformable GQA")
plt.xticks(tick_labels, labels=tick_labels)
plt.xlim([1, 32])
plt.xlabel("Deformable GQA Groups")
plt.ylabel("Runtime (ms)")
plt.grid(color="grey", linestyle="--")
plt.axhline(deformable_mqa_time, color="orange", linestyle="--", label="Deformable MQA")
plt.legend()
plt.show()

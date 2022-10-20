# +
import torch
from diffusion.models.unet import *

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

model = Unet(64)

model

model.forward(torch.randn(10, 3, 64, 64), torch.randint(0,1000, (10,)).long())

residual = Residual(torch.mean)
residual(torch.ones(10))

upsample = Upsample(64)
upsample(torch.randn(10, 64, 32, 32)).shape

downsample = Downsample(64)
downsample(torch.randn(10, 64, 32, 32)).shape

sin_emb = SinusoidalPosEmb(16)
sin_emb(torch.ones(10)).shape

block = Block(64, 32)
block(torch.randn(10, 64, 4, 4)).shape

res_block = ResnetBlock(dim=64, dim_out=32, time_emb_dim=16)
res_block(torch.randn(10, 64, 8, 8), time_emb = torch.ones(10, 16)).shape

linear_attn = LinearAttention(8)
print(linear_attn)
linear_attn(torch.ones(10, 8, 4, 4)).shape

attn = Attention(8)
print(attn)
attn(torch.ones(10, 8, 4, 4)).shape

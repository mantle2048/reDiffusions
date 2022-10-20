# +
import torch
import matplotlib.pyplot as plt
from diffusion.datas.dataset import *
from diffusion.datas.data_loader import *
from diffusion.datas.batch import Batch
from diffusion.user_config import DATASET_DIR
from torchvision.utils import make_grid
from einops import reduce, rearrange


# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

path = DATASET_DIR
print(path)

ds = Dataset(path, 96)

imgs = make_grid(torch.stack([ds[i] for i in range(10)]), nrow=10)
np_imgs = imgs.numpy()
fig = plt.figure()
plt.imshow(rearrange(np_imgs, 'c h w -> h w c'))

len(ds)

# +
config = {
    'dataset_path': DATASET_DIR,
    'image_size': 96,
    'batch_size': 256,
    
}
dl = make_dataloader(config)
# -

k = Batch(imgs = next(iter(dl)))

k.shape

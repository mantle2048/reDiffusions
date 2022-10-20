# +
from diffusion.models.ddpm_model import *
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import torch

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


# -

t = torch.arange(1000).long()
betas = linear_beta_schedule(1000)
fig = plt.figure()
plt.plot(t, betas)

t = torch.arange(1000).long()
betas = cosine_beta_schedule(1000)
fig = plt.figure()
plt.plot(t, betas)

config = {
    'image_size': 64,
    'objective': 'pred_noise',
    'noise_timesteps': 100,
    'beta_schedule': 'linear'
}
ddpm_model = GaussianDiffusion(config)

ddpm_model.predict_start_from_noise(
    x_t=torch.randn(10, 3, 64, 64),
    t=torch.randint(0,10, (10,)).long(),
    noise = torch.randn(10, 3, 64, 64)
).shape

ddpm_model.predict_noise_from_start(
    x_t=torch.randn(10, 3, 64, 64),
    t=torch.randint(0,10, (10,)).long(),
    x0 = torch.randn(10, 3, 64, 64)
).shape

posterior_mean, posterior_variance, posterior_log_variance_clipped = \
ddpm_model.q_posterior(
    x_start=torch.randn(10, 3, 64, 64),
    x_t = torch.randn(10, 3, 64, 64),
    t = torch.randint(0,10, (10,)).long(),
)
posterior_mean.shape

ModelPrediction = \
ddpm_model.model_predictions(
    x=torch.randn(10, 3, 64, 64),
    t = torch.randint(0,10, (10,)).long(),
)
ModelPrediction.pred_x_start.shape

model_mean, posterior_variance, posterior_log_variance, x_start = \
ddpm_model.p_mean_variance(
    x=torch.randn(10, 3, 64, 64),
    t = torch.randint(0,10, (10,)).long(),
)
model_mean.shape

pred_img, x_start = \
ddpm_model.p_sample(
    x=torch.randn(10, 3, 64, 64),
    t = 1,
)
fig = plt.figure()
show(make_grid(pred_img, nrow=10, normalize=True))

img = ddpm_model.p_sample_loop(
    shape=(10, 3, 64, 64)
)

fig = plt.figure()
show(make_grid(pred_img, nrow=10, normalize=True))

img = ddpm_model.q_sample(
    x_start = torch.randn(10, 3, 64, 64),
    t = torch.randint(0, 10, (10,)),
)
fig = plt.figure()
show(make_grid(img, nrow=10, normalize=True))

loss = ddpm_model.p_losses(
    x_start = torch.randn(10, 3, 64, 64),
    t = torch.randint(0, 10, (10,)),
)
print(loss)

train_log = ddpm_model.update(img = torch.randn(10, 3, 64, 64))
train_log



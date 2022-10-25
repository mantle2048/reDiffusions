import diffusion.infrastructure.utils.pytorch_util as ptu

from typing import Dict, Any
from itertools import cycle
from ema_pytorch import EMA
from diffusion.models import GaussianDiffusion
from diffusion.datas import make_dataloader, Batch

class DDPMAgent:

    def __init__(self, config: Dict):

        # init params
        self.config = config
        self.model = GaussianDiffusion(config['policy_config']).to(ptu.device)
        self.data_loader = cycle(make_dataloader(config['data_config']))
        self.lr_schedulers = None

        self.ema = EMA(self.model, beta = 0.995, update_every = 10)
        # TODO: Accelerator speed up accelerator_config

    def load_data(self) -> Batch:
        batch_images = next(iter(self.data_loader)).to(ptu.device)
        return Batch(images=batch_images)

    def train(self, batch: Batch) -> Dict:
        train_log = self.model.update(batch.images)
        self.ema.update()
        return train_log

    def sample(self, *args, **kwargs):
        self.ema.ema_model.eval()
        return self.ema.ema_model.sample(*args, **kwargs)

    def checkpoint(self, itr) -> Dict[str, Any]:
        data = {
            'itr': itr,
            'model': self.model.state_dict(),
            'optim': self.model.optim.state_dict(),
            'ema': self.ema.state_dict(),
        }
        return data

    def load(self, checkpoint_path):
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=ptu.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.optim.load_state_dict(checkpoint['optim'])
        self.ema.load_state_dict(checkpoint['ema'])


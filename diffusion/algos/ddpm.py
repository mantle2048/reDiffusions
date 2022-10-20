import diffusion.infrastructure.utils.pytorch_util as ptu

from typing import Dict, Any
from itertools import cycle
from diffusion.models import GaussianDiffusion
from diffusion.datas import make_dataloader, Batch

class DDPMAgent:

    def __init__(self, config: Dict):

        # init params
        self.config = config
        self.model = GaussianDiffusion(config['policy_config']).to(ptu.device)
        self.data_loader = cycle(make_dataloader(config['data_config']))
        self.lr_schedulers = None

        # TODO: EMA model
        # TODO: Accelerator speed up accelerator_config

    def load_data(self) -> Batch:
        imgs = next(iter(self.data_loader)).to(ptu.device)
        return Batch(imgs=imgs)

    def train(self, batch: Batch) -> Dict:
        train_log = self.model.update(batch.imgs)
        # self.ema_model.update()
        return train_log

    def sample(self, *args, **kwargs):
        # return self.ema_model.sample(*args, **kwargs)
        return self.model.sample(*args, **kwargs)

    def checkpoint(self, itr) -> Dict[str, Any]:
        data = {
            'itr': itr,
            'model': self.model.state_dict(),
            # 'ema': self.ema_model.state_dict(),
        }
        return data

    def load(self, checkpoint: Dict[str, Any]):
        pass

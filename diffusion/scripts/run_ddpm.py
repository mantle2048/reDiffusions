import hydra
import os.path as osp
from omegaconf import DictConfig, OmegaConf, open_dict
from diffusion.user_config import LOCAL_CONFIG_DIR, LOCAL_DATASET_DIR
from diffusion.algos.ddpm import DDPMAgent
from diffusion.diffusion_trainer import DiffusionTrainer

@hydra.main(version_base=None, config_path=LOCAL_CONFIG_DIR, config_name='ddpm')
def main(cfg: DictConfig):
    cfg.data_config.dataset_path = osp.join(LOCAL_DATASET_DIR, cfg.dataset)
    trainer = DiffusionTrainer(DDPMAgent, cfg)
    trainer.run_training_loop(cfg.n_itr)

if __name__ == '__main__':
    main()

    

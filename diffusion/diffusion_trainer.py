import random
import numpy as np
import torch
import time

from typing import Dict
from omegaconf import DictConfig, OmegaConf
from diffusion.infrastructure.loggers import setup_logger
from diffusion.infrastructure.utils import pytorch_util as ptu

class DiffusionTrainer:

    def __init__(self, agent_class, config: Dict):

        ######################
        ## INIT
        #####################

        self.config = config
        self.logger = setup_logger(**self.config['logger_config'])

        # Set random seed
        seed = self.config.setdefault('seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Init GPU
        ptu.init_gpu(
            use_gpu=not self.config['no_gpu'],
            gpu_id=self.config['which_gpu']
        )

        #############
        ## AGENT
        #############
        self.agent = agent_class(self.config)
        self.logger.log(f"Policy Net: {self.agent.model}", with_prefix=False)

    def run_training_loop(self, n_itr):
        """
        param n_itr:  number of iterations
        """
        # init vars at beginning of training
        self.start_time = time.time()

        for itr in range(1, n_itr + 1): #TODO Equiped with tqdm

            ## decide if tabular should be logged
            self._refresh_logger_flags(itr)

            ## load train data batch from DataLoader
            train_batch = self.agent.load_data()

            train_log = self.agent.train(train_batch)

            ###########################
            ## log and save config_json
            ###########################
            if itr == 1:
                self.logger.log_variant('config.yaml', self.config)

            ## log/save
            if self.logtabular:
                ## perform tabular and video
                self.perform_logging(itr, train_log)

            if self.logparam:
                self.logger.save_itr_params(itr, self.agent.get_weights())
                self.logger.save_extra_data(
                    self.agent.get_statistics(),
                    file_name='statistics.pkl',
                )
        self.logger.close()

    def perform_logging(self, itr, train_log: Dict):

        # sample and save images from ema model
        if self.logimage:
            print('\nSampling and saving images')
            # all_image = self.agent.sample(batch_size=self.config['num_samples'])
            # self.logger.log_images() #TODO
        #######################

        # save eval tabular
        if self.logtabular:
            # decide what to log
            self.logger.record_tabular("Itr", itr)
            self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)
            # TODO FID Score
            self.logger.record_dict(train_log)
            self.logger.dump_tabular(with_prefix=True, with_timestamp=False)

    def _refresh_logger_flags(self, itr):

        ## decide if videos should be rendered/logged at this iteration
        if self.config.get('image_log_freq', None) \
                and itr % self.config['image_log_freq'] == 0:
            self.logimage = True
        else:
            self.logimage = False

        if self.config.get('tabular_log_freq', None) \
                and itr % self.config['tabular_log_freq'] == 0:
            self.logtabular = True
        else:
            self.logtabular = False

        if self.config.get('param_log_freq', None) \
                and itr % self.config['param_log_freq'] == 0:
            self.logparam = True
        else:
            self.logparam = False

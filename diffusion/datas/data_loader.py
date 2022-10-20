from typing import Dict
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count

from diffusion.datas import Dataset

def make_dataloader(config: Dict) -> DataLoader:

    dataset_path = config['dataset_path']
    image_size = config['image_size']
    batch_size = config['batch_size']
    augment_horizontal_flip = config.get('augment_horizontal_flip', True)
    convert_image_to = config.get('convert_image_to', None)

    ds = Dataset(
        path = dataset_path,
        image_size = image_size,
        augment_horizontal_flip = augment_horizontal_flip,
        convert_image_to = convert_image_to
    )
    dl = DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        num_workers = cpu_count()
    )
    return dl

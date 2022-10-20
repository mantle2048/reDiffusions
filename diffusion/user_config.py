import os
import os.path as osp
cur_dir = os.getcwd()
par_dir = os.path.dirname(cur_dir)
LOCAL_LOG_DIR= osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
LOCAL_DIR = osp.abspath(osp.dirname(osp.dirname(__file__)))
LOCAL_DATASET_DIR = osp.join(LOCAL_DIR, '.dataset')
LOCAL_CONFIG_DIR = osp.join(LOCAL_DIR, 'config')

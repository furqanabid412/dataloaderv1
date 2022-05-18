import numpy as np


def class_mapping(labels):
    import yaml
    nusc_config = yaml.safe_load(open('configs/nusc/lidarseg/nusc_config.yaml', 'r'))
    learning_map = nusc_config['learning_map']

    lmap = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
    for k, v in learning_map.items():
        lmap[k] = v

    return lmap[labels]
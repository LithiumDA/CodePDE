import os
import sys

import hydra

from refine import refine
from repeated_sample import repeated_sample
from funsearch import funsearch

@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg):

    print(f'Method: {cfg.method.name}')
    print(f'Model name: {cfg.model.name}')
    print(f'PDE name: {cfg.pde.name}')
    
    print(f'Working folder: {cfg.working_folder}')
    if not os.path.exists(cfg.working_folder):
        os.makedirs(cfg.working_folder)
    if cfg.redirect_stdout:
        sys.stdout = open(os.path.join(cfg.working_folder, 'stdout.txt'), 'w')

    if cfg.method.name[:6] == 'refine':
        refine(cfg)
    elif cfg.method.name == 'repeated_sample':
        repeated_sample(cfg)
    elif cfg.method.name == 'funsearch':
        funsearch(cfg)
    else:
        raise NotImplementedError(f'Unknown method: {cfg.method.name}')

if __name__ == "__main__":
    main()

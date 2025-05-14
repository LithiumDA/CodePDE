import os
import sys
import time

from code_generation import generate_and_debug, prepare_working_folder

def repeated_sample(cfg):
    num_repeated_samples = cfg.method.num_repeated_samples
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name

    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    if cfg.redirect_stdout:
        sys.stdout = open(os.path.join(working_folder, 'stdout.txt'), 'w')

    print(f'Model name: {cfg.model.name}')
    print(f'Working folder: {working_folder}')

    prepare_working_folder(
        cfg, 
        working_folder=working_folder, 
        pde_name=pde_name,
        use_sample_solver_init=False
    )


    for sample_idx in range(num_repeated_samples):
        try:
            generate_and_debug(
                cfg,
                round_idx=sample_idx,
                num_trials=num_trials,
                pde_name=pde_name,
                working_folder=working_folder,
                seed_implementations=None,
                model_name=model_name
            )
        except Exception as e:
            print(f'Error in sample {sample_idx}: {e}. Move on to the next sample.')
        
        time.sleep(2)  # Small delay to prevent API rate limit

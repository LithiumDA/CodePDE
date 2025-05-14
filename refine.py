import os
import pandas as pd
import random
import time

from code_generation import generate_and_debug, prepare_working_folder

def select_seed_implementations(
    total_num_sample_solvers,
    num_sample_for_refine=None,
):
    if (
        num_sample_for_refine is None or 
        num_sample_for_refine > total_num_sample_solvers or
        num_sample_for_refine == -1
    ):
        num_sample_for_refine = total_num_sample_solvers

    # Select random samples for refinement
    selected_indices = random.sample(range(total_num_sample_solvers), num_sample_for_refine)
   
    return selected_indices
    


def refine(cfg):
    num_repeated_samples = cfg.method.num_repeated_samples
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name
    num_sample_for_refine = cfg.method.num_sample_for_refine
    start_round = cfg.method.start_round
    use_sample_solver_init = cfg.method.use_sample_solver_init
    assert use_sample_solver_init, 'Sample solvers must be enabled for refinement'

    sample_solver_folder = os.path.join(
        'solvers', pde_name, cfg.pde.pde_setting_name, 'seeds'
    )
    sample_solver_info = pd.read_csv(
        os.path.join(sample_solver_folder, 'seed_results.csv')
    )
    total_num_sample_solvers = len(sample_solver_info)

    if start_round == 0:
        prepare_working_folder(
            cfg, 
            working_folder=working_folder, 
            pde_name=pde_name,
            use_sample_solver_init=use_sample_solver_init
        )

    for round_idx in range(start_round, num_repeated_samples):
        try:
            seed_implementations = select_seed_implementations(
                total_num_sample_solvers=total_num_sample_solvers,
                num_sample_for_refine=num_sample_for_refine
            )
            generate_and_debug(
                cfg,
                round_idx=round_idx,
                num_trials=num_trials,
                pde_name=pde_name,
                working_folder=working_folder,
                seed_implementations=seed_implementations,
                model_name=model_name
            )
        except Exception as e:
            print(f'Error in sample {round_idx}: {e}. Move on to the next sample.')
        
        time.sleep(2)  # Small delay to prevent API rate limit

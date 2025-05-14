import math
import os
import pandas as pd
import random
import shutil
import time

from code_generation import generate_and_debug, prepare_working_folder, code_execution, get_results
from program_database import ProgramsDatabase, ProgramsDatabaseConfig
    
def get_seed_score(nRMSE, convergence_rate):
    return {
        'bucketed_convergence_rate': int(max(0, convergence_rate)*4),
        'bucketed_nRMSE': int(-math.log10(min(1e9, nRMSE))*10)
    }

def funsearch(cfg):
    num_trials = cfg.method.num_debugging_trials_per_sample
    pde_name = cfg.pde.name
    working_folder = cfg.working_folder
    model_name = cfg.model.name
    num_search_rounds = cfg.method.num_search_rounds
    num_initial_seeds = cfg.method.num_initial_seeds
    use_sample_solver_init = cfg.method.use_sample_solver_init
    assert use_sample_solver_init, 'Sample solvers must be enabled for refinement'

    sample_solver_folder = os.path.join(
        'solvers', pde_name, cfg.pde.pde_setting_name, 'seeds'
    )
    sample_solver_info = pd.read_csv(
        os.path.join(sample_solver_folder, 'seed_results.csv')
    )

    prepare_working_folder(
        cfg, 
        working_folder=working_folder, 
        pde_name=pde_name,
        use_sample_solver_init=use_sample_solver_init
    )

    pd_cfg = ProgramsDatabaseConfig()
    program_db = ProgramsDatabase(pd_cfg)

    # The first round: generate without seed
    seed_path = os.path.join(
        '../archived_logs', 
        pde_name, 
        cfg.pde.pde_setting_name,
        'repeated_sample',
        model_name
    )
    subdirectories = [d for d in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, d))]
    assert len(subdirectories) == 1, 'Only one subdirectory is expected'
    seed_path = os.path.join(seed_path, subdirectories[0])
    result_sheet = pd.read_csv(os.path.join(seed_path, 'test_results.csv'))

    for i in range(num_initial_seeds):
        relevant_files = [
            'errors_{idx}.txt',
            'implementation_{idx}.py',
            'output_{idx}.txt',
        ]

        complete_seed = True
        for file in relevant_files:
            if not os.path.exists(os.path.join(seed_path, file.format(idx=i))):
                complete_seed = False
                break
        if result_sheet[result_sheet['round'] == i].empty:
            complete_seed = False
        
        seed_info = result_sheet[result_sheet['round'] == i].to_numpy().tolist()[0]
        seed_info = [str(x) for x in seed_info]
        if seed_info[1] == 'failed' or seed_info[3] == 'nan':
            complete_seed = False

        if not complete_seed:
            continue

        # The seed is complete, copy it to the working folder
        for file in relevant_files:
            source_file = os.path.join(seed_path, file.format(idx=int(i)))
            destination_file = os.path.join(working_folder, file.format(idx=int(i)))
            shutil.copy(source_file, destination_file)
        with open(os.path.join(working_folder, 'test_results.csv'), 'a') as f:
            seed_info[0] = str(int(i))
            f.write(','.join(seed_info) + '\n')
        
        # Register the seed in the database
        seed_score = get_seed_score(float(seed_info[1]), float(seed_info[3]))
        with open(os.path.join(working_folder, f'implementation_{i}.py'), 'r') as f:
            implementation = f.readlines()
            program_len = len(implementation)
        program_db.register_program(
            program=i,
            program_len=program_len,
            island_id=None,
            scores_per_test=seed_score,
        )

    for i in range(num_initial_seeds, num_initial_seeds+num_search_rounds):
        island_id, seed_ids = program_db.get_seed()
        try:
            relative_error, elapsed_time, avg_rate = generate_and_debug(
                cfg,
                round_idx=i,
                num_trials=num_trials,
                pde_name=pde_name,
                working_folder=working_folder,
                seed_implementations=seed_ids,
                model_name=model_name
            )
            seed_score = get_seed_score(float(relative_error), float(avg_rate))
            with open(os.path.join(working_folder, f'implementation_{i}.py'), 'r') as f:
                implementation = f.readlines()
                program_len = len(implementation)
            program_db.register_program(
                program=i,
                program_len=program_len,
                island_id=island_id,
                scores_per_test=seed_score,
            )
        except Exception as e:
            print(f'Error in round {i}: {e}. Move on to the next sample.')
        

    # Finally, report the best program
    results = pd.read_csv(os.path.join(working_folder, 'test_results.csv'))
    keywords = ['nRMSE', 'elapsed_time', 'convergence_rate']
    for keyword in keywords:
        results[keyword] = pd.to_numeric(results[keyword], errors="coerce")
    # Sort by nRMSE, elapsed_time, and convergence_rate
    sorted_results = results.sort_values(by=keywords, ascending=[True, True, False])
    best_idx = int(sorted_results.head(1)["round"].values[0])
    
    test_run_id = 999
    shutil.copy(
        os.path.join(working_folder, f'implementation_{best_idx}.py'),
        os.path.join(working_folder, f'implementation_{test_run_id}.py')
    )
    execution_results = code_execution(
        cfg,
        working_folder = working_folder,
        round_idx=test_run_id,
        pde_name=pde_name,
        eval_dataset=os.path.join(
            cfg.root_dataset_folder, 
            cfg.pde.dataset_folder_for_eval.replace('_development.hdf5', '.hdf5')
        )
    )

    if execution_results['exit_code'] != 0:
        relative_error, elapsed_time, avg_rate = None, None, None
    else:
        relative_error, elapsed_time, avg_rate = get_results(
            os.path.join(working_folder, f'output_{test_run_id}.txt')
        )
    with open(os.path.join(working_folder, 'final_result.txt'), 'w') as f:
        f.write('best_idx,relative_error,elapsed_time,avg_rate\n')
        f.write(f'{best_idx},{relative_error},{elapsed_time},{avg_rate}\n')

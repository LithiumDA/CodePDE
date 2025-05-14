import json
import math
import os
import re
import shutil
import signal
import subprocess
import time


from llm_api import generate_response
from prompt_files import general_prompt, pde_descriptions


def file_to_string(file_path):
    with open(file_path) as f:
        string = ''.join(f.readlines())
    return string


def get_last_line(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    result_line = lines[-1]
    return result_line


def generate_pde_description(cfg, pde_name):
    if pde_name == 'advection':
        pde_description = pde_descriptions.advection_description.format(advection_beta=cfg.pde.beta)
    elif pde_name == 'burgers':
        pde_description = pde_descriptions.burgers_description.format(burgers_nu=cfg.pde.nu)
    elif pde_name == 'reacdiff1d':
        pde_description = pde_descriptions.reacdiff_1d_description.format(reacdiff1d_nu=cfg.pde.nu,
            reacdiff1d_rho=cfg.pde.rho)
    elif pde_name == 'cns1d':
        pde_description = pde_descriptions.cns1d_description.format(cns1d_eta=cfg.pde.eta)
    elif pde_name == 'darcy':
        pde_description = pde_descriptions.darcy_description.format()
    elif pde_name == 'ins2d':
        pde_description = pde_descriptions.ins2d_description.format()
    else:
        raise ValueError(f'PDE {pde_name} not recognized')
    return pde_description


def generate_initial_prompt_without_seed(cfg, pde_name):
    system_prompt = general_prompt.system_prompt
    pde_description = generate_pde_description(cfg, pde_name)
    
    solver_template = file_to_string(f'solvers/{pde_name}/solver_template.py')

    problem = general_prompt.code_generation_without_seed_prompt.format(
        pde_description=pde_description,
        solver_template=solver_template
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    return messages 


def generate_initial_prompt(
    cfg,
    seed_implementations:list,
    working_folder: str,
    pde_name:str = 'burgers'
):
    system_prompt = general_prompt.system_prompt

    pde_description = generate_pde_description(cfg, pde_name)

    if cfg.method.name == 'funsearch':
        seed_folder = working_folder
    else:
        # cfg.method.name == 'refine'
        seed_folder = os.path.join('solvers', pde_name, cfg.pde.pde_setting_name, 'seeds')
    examples = [
        general_prompt.code_sample.format(
            id=example_id,
            code=file_to_string(os.path.join(seed_folder, f'implementation_{seed_id}.py')),
            code_output=get_last_line(os.path.join(seed_folder, f'output_{seed_id}.txt')),
        )
        for example_id, seed_id in enumerate(seed_implementations)
    ]
       
    code_samples = ''.join(examples)
    
    problem = general_prompt.problem_prompt.format(
        pde_description=pde_description,
        code_samples=code_samples)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    return messages


def generate_debugging_prompt(
    round_idx:int,
    working_folder: str,
    debugging_reason:str = 'execution_error'
):
    # Load the prompt from the file
    with open(os.path.join(working_folder, f'messages_{round_idx}.json'), 'r') as f:
        messages = json.load(f)
    # Load model response
    model_response = file_to_string(os.path.join(working_folder, f'responses_{round_idx}.txt'))
    messages.append({"role": "assistant", "content": model_response})
    # Load the error message (truncated to the last 5000 characters)
    code_output = file_to_string(os.path.join(working_folder, f'output_{round_idx}.txt'))[-5000:]
    errors = file_to_string(os.path.join(working_folder, f'errors_{round_idx}.txt'))[-5000:]
    if debugging_reason == 'execution_error':
        feebdack = general_prompt.debugging_execution_error_prompt.format(
            code_output=code_output,
            error_message=errors
        )
    else: # debugging_reason == 'nan_inf'
        feebdack = general_prompt.debugging_nan_inf_prompt.format(
            code_output=code_output,
            error_message=errors
        )
    messages.append({"role": "user", "content": feebdack})
    return messages


def generate_prompt(
    cfg,
    round_idx:int,
    working_folder: str,
    seed_implementations: list|None = None,
    generation_mode:str='initial',
    pde_name:str='burgers'
):
    if generation_mode == 'debugging_execution_error':
        prompt = generate_debugging_prompt(
            round_idx=round_idx,
            working_folder=working_folder,
            debugging_reason='execution_error'
        )
    elif generation_mode == 'debugging_nan_inf':
        prompt = generate_debugging_prompt(
            round_idx=round_idx,
            working_folder=working_folder,
            debugging_reason='nan_inf'
        )
    elif seed_implementations is None or len(seed_implementations) == 0:
        prompt = generate_initial_prompt_without_seed(
            cfg,
            pde_name=pde_name
        )
    else:
        prompt = generate_initial_prompt(
            cfg,
            seed_implementations=seed_implementations,
            working_folder=working_folder,
            pde_name=pde_name
        )

    return prompt


def code_generation(
    cfg, 
    round_idx:int,
    working_folder: str,
    seed_implementations: list|None = None,
    generation_mode: str = 'initial',
    pde_name: str = 'burgers',
    model_name='deepseek-chat'
):

    messages = generate_prompt(
        cfg,
        round_idx=round_idx,
        working_folder=working_folder,
        seed_implementations=seed_implementations,
        generation_mode=generation_mode,
        pde_name=pde_name
    )

    # Save the messages to a file
    with open(os.path.join(working_folder, f'messages_{round_idx}.json'), 'w') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4) 
    responses = generate_response(messages, cfg)
    if 'claude' in model_name:
        content = ''
        for block in responses.content:
            if block.type == 'thinking':
                # Save the CoT of Claude-thinking
                with open(os.path.join(working_folder, f'thinking_{round_idx}.txt'), 'w') as f:
                    f.write(str(block.thinking))
                if content == '':
                    content = block.thinking
            elif block.type == 'text':
                # Extract the final response
                content = block.text
    elif 'gemini' in model_name:
        content = responses.text
    elif 'qwq' in model_name:
        content = responses
    else:
        content = responses.choices[0].message.content
    # Save the response to a file
    with open(os.path.join(working_folder, f'responses_{round_idx}.txt'), 'w') as f:
        f.write(content)

    matches = re.findall(
        r'```python(.*?)```',
        content, re.DOTALL)

    if not matches:
        raise ValueError('No relevant code block found in response')

    generated_code = max(matches, key=len)

    with open(os.path.join(working_folder, f'implementation_{round_idx}.py'), 'w') as f:
        f.write(generated_code)


def code_execution(
    cfg, 
    working_folder: str,
    round_idx: int = 0,
    pde_name: str = 'burgers',
    eval_dataset: str = None
):
    # Copy the implementation file to solver.py to make the evaluator's life easier
    os.system(f'cp {working_folder}/implementation_{round_idx}.py {working_folder}/solver.py')
    
    # Open files for standard output and error logging
    job_out = open(os.path.join(working_folder, f'output_{round_idx}.txt'), 'w')
    job_err = open(os.path.join(working_folder, f'errors_{round_idx}.txt'), 'w')

    # Construct the base command
    if eval_dataset is None:
        eval_dataset = os.path.join(cfg.root_dataset_folder, cfg.pde.dataset_folder_for_eval)
    cmd = (
        f'CUDA_VISIBLE_DEVICES={cfg.assigned_gpu} '
        f'python {working_folder}/evaluator.py '
        f'--save-pth {working_folder} '
        f'--run-id {round_idx} '
        f'--dataset-path-for-eval '
        f'{eval_dataset} '
    )
    # Note: In Funsearch, we will need to customize the eval_dataset to seperate development and testing

    # Append PDE-specific hyperparameters to the command
    if pde_name == 'advection':
        hyperparam = f'--beta {cfg.pde.beta} '
    elif pde_name == 'burgers':
        hyperparam = f'--nu {cfg.pde.nu} '
    elif pde_name == 'reacdiff1d':
        hyperparam = f'--nu {cfg.pde.nu} --rho {cfg.pde.rho} '
    elif pde_name == 'cns1d':
        hyperparam = f'--eta {cfg.pde.eta} '
    elif pde_name in ['darcy', 'ins2d']:
        hyperparam = f' '  # No hyperparameters for these two
    else:
        raise ValueError(f'PDE {pde_name} not recognized')
    
    try:
        # Start process using Popen
        process = subprocess.Popen(
            f'{cmd} {hyperparam}',
            shell=True,
            stdout=job_out,
            stderr=job_err,
            text=True,
            preexec_fn=os.setsid  # Create a new process group
        )
        
        # Wait for the process with timeout
        exit_code = process.wait(timeout=cfg.pde.timeout)
        stderr = None
        status = "completed"
  
    except subprocess.TimeoutExpired:
        # Kill the entire process group on timeout
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        # Wait a moment for graceful termination
        time.sleep(2)
        
        # If still running, use SIGKILL
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        
        # Clean up any GPU processes that might still be running
        cleanup_gpu_processes(cfg.assigned_gpu)
        
        job_out.write(f"Process exceeded the {cfg.pde.timeout}-second timeout limit.\n")
        job_err.write(f"Process exceeded the {cfg.pde.timeout}-second timeout limit.\n")
        exit_code = -1
        stderr = "TimeoutExpired: Process exceeded the timeout limit."
        status = "timeout"
        
    finally:
        # Always close the files
        job_out.close()
        job_err.close()

    return {
        "exit_code": exit_code,
        "stderr": stderr,
        "status": status
    }

def cleanup_gpu_processes(gpu_id):
    """
    Clean up any orphaned processes still using the specified GPU
    """
    try:
        # Find all processes using this GPU
        result = subprocess.run(
            f"nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i {gpu_id}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Extract process IDs
        pids = result.stdout.strip().split('\n')
        
        # Kill each process
        for pid in pids:
            if pid and pid.isdigit():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Killed GPU process with PID {pid}")
                except ProcessLookupError:
                    pass  # Process already terminated
    except Exception as e:
        print(f"Error during GPU cleanup: {e}")


def get_results(output_file):
    result_line = get_last_line(output_file)

    relative_error_match = re.search(r'nRMSE: (.*?)\t', result_line)
    relative_error = float(relative_error_match.group(1))

    elapsed_time_match = re.search(r'Time: (.*?)s', result_line)
    elapsed_time = float(elapsed_time_match.group(1))

    avg_rate_match = re.search(
        r'Average convergence rate: (.*?)\t', result_line)
    avg_rate = float(avg_rate_match.group(1))

    return relative_error, elapsed_time, avg_rate


def prepare_working_folder(
    cfg, 
    working_folder, 
    pde_name='burgers',
    use_sample_solver_init=False
):
    result_sheet_path = os.path.join(working_folder, 'test_results.csv')
    print('Generating result sheet')
    with open(result_sheet_path, 'w') as f:
        f.write('round,nRMSE,elapsed_time,convergence_rate,num_trial\n')

    evluator_path = os.path.join(working_folder, f'evaluator.py')
    os.system(f'cp solvers/{pde_name}/evaluator.py {evluator_path}')
    
    if use_sample_solver_init:
        # We don't copy the sample solvers, nor execute them.
        pass


def generate_and_debug(
    cfg,
    round_idx:int,
    num_trials:int,
    pde_name:str,
    working_folder:str,
    seed_implementations:list|None,
    model_name:str
):
    generation_mode = 'initial'
    for num_trial in range(1, num_trials+1):
        # When num_trial==1, it is not debugging
        # The output of the generated code will be saved in 
        # os.path.join(working_folder, f'generated_code_{round_idx}.txt')
        code_generation(
            cfg, 
            round_idx=round_idx,
            working_folder=working_folder,
            seed_implementations=seed_implementations,
            generation_mode=generation_mode,
            pde_name=pde_name,
            model_name=model_name
        )
        print(f'Round {round_idx}, trial {num_trial} code generation completed successfully')

        print(f'Round {round_idx}, trial {num_trial} code execution started')
        execution_results = code_execution(
            cfg, 
            working_folder=working_folder,
            round_idx=round_idx,
            pde_name=pde_name
        )

        if execution_results['exit_code'] != 0:
            print(f'Error in round {round_idx}, trial {num_trial} code execution.')
            if num_trial < num_trials:
                print(f'Let LLM debug the code')
                generation_mode = 'debugging_execution_error'
            else:
                with open(os.path.join(working_folder, 'test_results.csv'), 'a') as f:
                    f.write(f'{round_idx},failed,failed,failed,{num_trial}\n')
                raise ValueError(f'Error in round {round_idx}, trial {num_trial} code execution.')
            
        else:
            print(f'Round {round_idx}, trial {num_trial} completed successfully')
            relative_error, elapsed_time, avg_rate = get_results(
                os.path.join(working_folder, f'output_{round_idx}.txt')
            )

            if (
                (math.isnan(relative_error) or math.isinf(relative_error))
                and num_trial < num_trials
            ):
                # If we get NaN or Inf in nRMSE and still have chances to debug, we will debug the code
                print(f'nRMSE is NaN/Inf in round {round_idx}, trial {num_trial} code execution.')
                print(f'Let LLM debug the code')
                generation_mode = 'debugging_nan_inf'
            else:
                # Otherwise, we will save the results and break the loop
                with open(os.path.join(working_folder, 'test_results.csv'), 'a') as f:
                    f.write(f'{round_idx},{relative_error},{elapsed_time},{avg_rate},{num_trial}\n')
                print(f'nRMSE: {relative_error:.5f}\t| Time: {elapsed_time:.2f}s\t| Rate: {avg_rate}\t| Trial: {num_trial}')
                return relative_error, elapsed_time, avg_rate
    return None, None, None

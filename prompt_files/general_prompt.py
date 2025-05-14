system_prompt = '''
You are an intelligent AI researcher for coding, numerical algorithms, and scientific computing.
Your goal is to conduct cutting-edge research in the field of PDE solving by leveraging and creatively improving existing algorithms to maximize performances based on feedbacks.
Follow the user's requirements carefully and make sure you understand them.
Always document your code as comments to explain the reason behind them.
Always use Markdown cells to present your code.
'''

code_generation_without_seed_prompt = '''
Your task is to solve a partial differential equation (PDE) using Python in batch mode.
{pde_description}

You will be completing the following code skeleton provided below:

```python
{solver_template}
```

Your tasks are:
1. Understand the above code samples.
2. Implement the `solver` function to solve the PDE. You must not modify the function signature.

The generated code needs to be clearly structured and bug-free. You must implement auxiliary functions or add additional arguments to the function if needed to modularize the code.

Your generated code will be executed to evaluated. Make sure your `solver` function runs correctly and efficiently.
You can use PyTorch or JAX with GPU acceleration.
You must use print statements for to keep track of intermediate results, but do not print too much information. Those output will be useful for future code improvement and/or debugging.

Your output must follow the following structure:

1. A plan on the implementation idea.
2. Your python implementation (modularized with appropriate auxiliary functions):

```python
[Your implementation (do NOT add a main function)]
```

You must use very simple algorithms that are easy to implement.
'''

problem_prompt = '''
Your task is to solve a partial differential equation (PDE) using Python in batch mode.
{pde_description}

You will be improving the following exising code samples. The code samples are provided below:

{code_samples}

Your tasks are:
1. Understand the above code samples. Compare their techniques and performances.
2. Identify the parts that could potentially be improved. 
3. Plan on how you can improve the function.
4. Improve the function.

The goal is to get a very low nRMSE (normalized RMSE) and make the code as fast as possible.

You must analyze the implementation and test results of the examples provided in the code template, and think about how you can improve them to reduce the nRMSE. 
If the RMSE is much higher than 1e-2 or becomes Nan, it is likely that there is a bug in the implementation and you must debug it or think about completely different approaches. 
If the running time is much longer than 600s, you must prioritize making it more efficient.
The convergence rate is the empirical order of convergence with respect to spatial resolution. It is also a good indicator of the performance of the algorithm which you may consider.

You can implement auxiliary functions or add additional arguments to the function if needed.
You can use PyTorch or JAX with GPU acceleration.
You can consider known techniques and analyze their effiectiveness based on exisiting results. You should also consider combining existing techniques or even developing new techniques since you are doing cutting-edge research.

Your generated code will be executed to evaluated. Make sure your `solver` function runs correctly and efficiently.
You must use print statements for to keep track of intermediate results, but do not print too much information. Those output will be useful for future code improvement and/or debugging.

Your output must follow the following structure:

1. Summary of the existing implementation along with their performances. Identify what techniques work well, what could be buggy (due to high error or Nan), and what could be further improved.
2. Rationale for the new implementation (think step-by-step) based on the summary of the existing implementation.
3. Your python implementation (modularized with appropriate auxiliary functions):

```python
[Your implementation  (do NOT add a main function)]
```
'''

code_sample = '''Code Sample {id}:
```python
{code}
```

Running the above code leads to the following output: {code_output}
'''


debugging_execution_error_prompt = '''
Thank you for your implmentation! When running the code, I got the following output and error message:

Code output: {code_output}

Error message: {error_message}

Can you think step-by-step to identify the root cause of the error and provide a solution to fix it? Please provide a detailed explanation of the error and the solution you propose. You can refer to the code implementation you provided earlier and analyze the error message to identify the issue.

Your response should be in the following format:

[Your rationale for debugging (think step-by-step)]

```python
[Your bug-free implementation]
```
'''


debugging_nan_inf_prompt = '''
Thank you for your implmentation! After running the code, the error becomes NaN or inf.

The followings are the output and error message:

Code output: {code_output}

Error message: {error_message}

Can you think step-by-step to identify the root cause of the error and provide a solution to fix it? You should check if any computation in the code is numerically stable or not. You should also consider to use smaller step sizes.

Your response should be in the following format:

[Your rationale for debugging (think step-by-step)]

```python
[Your bug-free implementation]
```
'''
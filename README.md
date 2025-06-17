## CodePDE

This is the official implementation for [CodePDE: An Inference Framework for LLM-driven PDE Solver Generation](https://arxiv.org/abs/2505.08783), the first inference framework for generating PDE solvers using large language models (LLMs).

### Dependencies

The required packages are listed in `requirements.txt`, which can be installed by running `pip install -r requirements.txt`. 

### Getting started

Download the data using the `data/data_download.py` script from PDEbench. Then process the data with the `data/extract_data_subsets.py` script.

Set up the configurations in `config` and run `python main.py`.

In the _repeated sampling_ mode, the LLM generates solvers from scratch.

In the _refinement_ mode, the LLM uses existing solvers in the `solvers` folder as "seeds" (e.g., `solvers/burgers/nu_0.01/seeds` for Burgers Equation with $\nu=0.01$) and tries to improve upon the "seeds".

In the _funsearch_ mode, the LLM uses a few solvers generated in the _repeated sampling_ stage to warm start the program database and then generates new solvers via evolutionary search. The implementation assumes that the _repeated sampling_ results are stored under `../archived_logs`.

### Contact

May you have any questions on our work or implementation, feel free to reach out to [`shandal@cs.cmu.edu`](shandal@cs.cmu.edu)!

If you find this repository useful, please consider giving a star ⭐ and cite our paper.

```
@article{li2025codepde,
  title={CodePDE: An Inference Framework for LLM-driven PDE Solver Generation},
  author={Li, Shanda and Marwah, Tanya and Shen, Junhong and Sun, Weiwei and Risteski, Andrej and Yang, Yiming and Talwalkar, Ameet},
  journal={arXiv preprint arXiv:2505.08783},
  year={2025}
}
```
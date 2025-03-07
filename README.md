# Robust in-context learning for model-free system identification

<!-- This repository contains the Python code to reproduce the results of the paper [In-context learning for model-free system identification](http://arxiv.org/abs/2308.13380)
by Matteo Rufolo, Dario Piga, Marco Forgione. -->

In this repository we address the robustness characteristic in the model-free in-context learning architecture introduced here [In-context learning for model-free system identification](https://arxiv.org/abs/2410.03291) for System Identification, where a *meta model* is trained to describe an entire class of dynamical systems.
We focus on analyzing the same optimization problem, but from a task distribution perspective, prioritizing tasks with higher losses during training, thereby focusing on the most potentially problematic cases.

<!-- 
## Minimization problem

With respect to the original paper we analyze a new optimization problem:


$$\phi^* = \arg \min_{\phi} \mathbb{E}_{p_\alpha(\mathcal{D}, \phi)}\left[-\log q_{\theta(\phi,X)}(\tilde{y}_{\nin+1:n})\right]$$ -->

## Architecture

We employ the same Transformer architecture of the original paper:

<!-- ![machine-translation-like model-free simulation](fig/encoder_decoder_architecture.png "Generalized multi-step-ahead simulation") -->
<img src="figure/encoder_decoder_architecture.png"  width="1400">

# Main files

We have been performed 2 different training on the WH system class, and their scripts are:

* [train_sim_WH_active_top.py](train_sim_WH_active_top.py), that is the the training of the original paper
* [train_sim_WH_active_rand.py](train_sim_WH_active_rand.py), that is the robust training procedure proposed in this repository

The script above accept command-line arguments to customize the architecture and aspects of the training. 
The goal of this repository is to imrpove the robustness of the work presented in [sysid-transformer](https://github.com/mattrufolo/sysid-prob-transformer) analyzing exactly those dataset that the Transformer is more uncertain about.

Trained weights of all the Transformers discussed in the example section of the paper are available as assets in the [v0.1 Release](https://github.com/mattrufolo/sysid-robust-transformer/releases/tag/v0.1) for all the random inizializations of the meta model and datasets realizations.



Jupyter notebooks that load the trained models and make predictions/simulations on new data generated by the WH system class is also available in the notebook file [test_sim_wh_active.ipynb](test_sim_wh_active.ipynb), while the performance of the tranined meta models on the SilverBox benchmark is available in the notebook file [test_sim_wh_active_bench.ipynb](test_sim_wh_active_bench.ipynb)

# Software requirements
Experiments were performed on a Python 3.11 conda environment with:

 * numpy
 * scipy
 * pandas
 * matplotlib
 * tdqm
 * numba
 * copy
 * nonlinear_benchmarks
 * pytorch (v2.2.2)

These dependencies may be installed through the commands:

```
conda install numpy scipy matplotlib tdqm numba pandas copy nonlinear_benchmarks
conda install pytorch -c pytorch
```

For more details on pytorch installation options (e.g. support for CUDA acceleration), please refer to the official [installation instructions](https://pytorch.org/get-started/locally/).

The following packages are also useful:

```
conda install jupyter # (optional, to run the test jupyter notebooks)
pip install wandb # (optional, for experiment logging & monitoring)
```

# Hardware requirements
While all the scripts can run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a server equipped with an nVidia RTX 3090 GPU.


# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 



<!-- * Cite the [paper](https://arxiv.org/abs/2308.13380) 
```
@article{forgione2023from,
  author={Forgione, Marco and Pura, Filippo and Piga, Dario},
  journal={IEEE Control Systems Letters}, 
  title={From System Models to Class Models:
   An In-Context Learning Paradigm}, 
  year={2023},
  volume={7},
  number={},
  pages={3513-3518},
  doi={10.1109/LCSYS.2023.3335036}
}
``` -->
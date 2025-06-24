# DRETune: Accelerating Database Parameter Tuning Under Sample Deficiency
This repository contains the source code for the paper "DRETune: Accelerating Database Parameter Tuning
Under Sample Deficiency" . DRETune, a tuning framework that combines zero-shot
dimensionality reduction, expert knowledge, and deep
reinforcement learning to enhance tuning performance
with limited samples

DRETune utilizes a
dual random projection matrix technique for sample-
free parameter dimensionality reduction, mitigating
the negative impact of low-dimensional projections on
tuning outcomes. Recognizing the limitations of expert
experience in guiding knob adjustments, the tuning
process is innovatively divided into two stages: trend
adjustment and magnitude adjustment.

## Source Code Structure

- `hes/`

    `low_dim_adaptor.py:`The main implementation function for dimensionality reduction using double random projection matrices.
    `num_mapper.py:`Solving the problem of projecting multiple important dimensions onto the same low - dimensional space.
- `maEnv/*:`Node environment instances and tool instances.
- `model_transfer/`

  `load_feature.py:`Obtain feature vectors.

  `make_model.py:` Code related to creating datasets required for transfer.
  
- `my_algorithm/` Algorithm implementation.
  
  `agent.py:` Implementation of the agent.
  
  `sac_2.py/sac_model.py:` Construction of SAC algorithm network models.

  `replay_memory.py:` Implementation of the experience pool.

  `LERC.py:` Code related to the generation of rule - based simulated experience.

- `test_scripts/*`Some test scripts.

- `transfer_rpm`The main code for transfer learning in transfer_rpm.

NOTE: This code has been adapted to the internal database to some extent, as it needs to obtain a lot of internal database information that cannot be obtained through SQL statements. If you want to use this tuning code, you can modify the communication with the database.
## Environment

We chose an InnoDB-
based CSDB cluster, which emulates Amazon Aurora’s
decoupled storage-compute architecture , for our
experiments. CSDB consists of a master compute node (MCN),
slave compute nodes (SCN), and storage nodes (SN), all
connected via network communication.

The source code is recommended to run under the Python 3.8 environment.. To install the required packages run the following command:

    pip3 install -r requirements.txt    
## Run Experiments

Training Entry main.py

Command to start the training process: python main.py --algorithm SAC_2

Run in the background: nohup python -u main.py --algorithm SAC_2 > {log_dir_path}/log_{name}_{date}.log &
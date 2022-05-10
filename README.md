# Survival MDN

This repo provides a pytorch implementation for the proposed model, *Survival MDN*, in the following paper:

[Survival Mixture Density Networks]

The code is built based on the repo https://github.com/jiaqima/SODEN

## Requirements

Most required libraries should be included in `environment.yml`. To prepare the environment, run the following commands:
```shell
conda env create -f environment.yml
conda activate soden
```

## Datasets
The 10 splits of SUPPORT, METABRIC and GBSG datasets are available at `data/support`, `data/metabric` and `data/gbsg` respectively.

The MIMIC datasets are derived from the [MIMIC-IV database](https://mimic.physionet.org/gettingstarted/access/), which requires individual licenses to access. The query of the preprocessing is attached in the paper.

## Usage
### Generate config files
1. The folder `configs/` includes model config templates for various models. `range_specs.py` defines tuning ranges of hyper-parameters.

2. Run `generate_config.py` to randomly generate complete hyper-parameter configurations for random search. For example, to generate 100 random configurations for the proposed Survival MDN model on SUPPORT data, run the following command:
```shell
python generate_config.py --basic_model_config_file configs/support.json --num_trials 100 --starting_trial 0 --random_seed 0
```
The generated 100 complete hyper-parameter configuration files will be stored in the folder `data/hp_configs`.

### Train
Example command to run a single trial of training Survival MDN on the SUPPORT data with the split 1 (out of 1-10):
```shell
python main.py --save_model --save_log --dataset support --path data/support --split 1 --seed 1 --model_config_file data/hp_configs/support__0__model.json --train_config_file data/hp_configs/support__0__train.json
```
In particular, the model and training hyper-parameters are specified by the files `support__0__model.json` and `support__0__train.json` we generated before. 

A model config filename takes the form `<dataset>__<trial_id>__model.json` and a training config filename takes the form `<dataset>__<trial_id>__train.json`.

### Evaluation
A couple of evaluation metrics including the loss function will be calculated throughout the training procedure. But some evaluation metrics are time-consuming so their calculations are left to a dedicated evaluation mode. After training is completed, run the following command to evaluate.
```shell
python main.py --evaluate --dataset support --path data/support --split 1 --seed 1 --model_config_file data/hp_configs/support__0__model.json --train_config_file data/hp_configs/support__0__train.json
```
This command will load the best model checkpoint and make the evaluation (make sure the `--save_model` argument is added to the training command).


## Acknowledgement
Thanks Weijing Tang, Jiaqi Ma, Qiaozhu Mei, and Ji Zhu for providing the codebase https://github.com/jiaqima/SODEN 

Thanks Weijing Tang for detailed explanation of the codebase.

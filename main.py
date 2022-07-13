from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from data import get_mimic_dataloader, get_mimic_seq_dataloader
from metrics import (BinomialLogLikelihoodMeter, BrierScoreMeter, CIndexMeter,
                     IPWCIndexMeter, ConcordanceMeter, IPWConcordanceMeter,
                     QuantileConcordanceMeter)
from mdn_models import MDNModel
from six.moves import cPickle as pickle
from trainers import MDNTrainer
from utils import SEP

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--dataset", default="support")
parser.add_argument("--path", default="./data/support/")
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--fine_tune", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--seed", type=int, default=-1)

# Dataset configuration
parser.add_argument("--split", type=int, default=1)

# Model configuration.
parser.add_argument(
    "--model_config_file",
    default="./configs/support__rec_mlp__0__model.json",
    help="Suggested format: Dataset_name__model_type__trial_id__model.json")

# Training configuration.
parser.add_argument(
    "--train_config_file",
    default="./configs/support__rec_mlp__0__train.json",
    help="Suggested format: Dataset_name__model_type__trial_id__train.json")

# Other configuration
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--result_path", default=None)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--model_path", default=None)
parser.add_argument("--save_log", action="store_true")
parser.add_argument("--log_path", default=None)
parser.add_argument("--save_raw", action="store_true")

args = parser.parse_args()

# Set random seed
if args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

# Load train config.
with open(args.train_config_file) as f:
    train_config = json.load(f)

use_full_size_per_batch = False

# Prepare the dataset.
dataloaders = {}
filenames = {}
feature_size = None
use_embed = False

# Get Dataset
assert args.path.strip("/").split("/")[-1] == args.dataset
# Prepare dataloaders.
random_state = np.random.RandomState(seed=0)

for phase in ["train", "valid", "test"]:
    if args.fine_tune and phase in ["train", "valid"]:
        input_file = os.path.join(
            args.path, phase + "_%d_fine_tune.npz" % args.split)
    else:
        input_file = os.path.join(
            args.path, phase + "_%d.npz" % args.split)
        filenames[phase] = input_file
    batch_size = train_config["batch_size"]
    if use_full_size_per_batch and phase == "train":
        dt = np.load(input_file)
        batch_size = dt["arr_1"].shape[0]
    elif phase != "train":
        batch_size = 1024
    dataloaders[phase], feature_size = get_mimic_dataloader(
        input_file,
        batch_size,
        random_state,
        is_eval=(phase != "train"))


# Initialize the model.
# Load model config.
with open(args.model_config_file) as f:
    model_config = json.load(f, object_pairs_hook=OrderedDict)
model = MDNModel(model_config=model_config, feature_size=feature_size)
model.to(args.device)

# Survival loss.
def survival_loss(outputs, labels):
    if torch.isnan(torch.abs(outputs["lambda"]).max()):
        sys.exit()
    batch_loss = -labels * torch.log(
        outputs["lambda"].clamp(min=1e-8)) + outputs["Lambda"]
    return torch.mean(batch_loss)


class SurvivalLossMeter(object):
    def __init__(self):
        super(SurvivalLossMeter, self).__init__()
        self.reset()

    def add(self, outputs, labels):
        self.values.append(survival_loss(outputs, labels).item())

    def value(self):
        return [np.mean(self.values)]

    def reset(self):
        self.values = []


# Optimization criterions.
criterions = {}
criterions["survival_loss"] = survival_loss
# Optimizer.
optimizer = optim.RMSprop(
    model.parameters(),
    lr=train_config["learning_rate"],
    weight_decay=train_config["weight_decay"])



# Evaluation metrics.
metrics = {}
metrics["survival_loss"] = SurvivalLossMeter()
if args.evaluate:
    # metrics["concordance_time_dependent"] = CIndexMeter(save_raw=args.save_raw)
    metrics["ipw_concordance_time_dependent"] = IPWCIndexMeter()
    metrics["ipw_2_concordance_time_dependent"] = IPWCIndexMeter(eps=0.2)
    metrics["ipw_4_concordance_time_dependent"] = IPWCIndexMeter(eps=0.4)
    metrics["brier_score"] = BrierScoreMeter()
    metrics["brier_score_2"] = BrierScoreMeter(eps=0.2)
    metrics["brier_score_4"] = BrierScoreMeter(eps=0.4)
    metrics["binomial_log_likelihood"] = BinomialLogLikelihoodMeter()
    metrics["binomial_log_likelihood_2"] = BinomialLogLikelihoodMeter(eps=0.2)
    metrics["binomial_log_likelihood_4"] = BinomialLogLikelihoodMeter(eps=0.4)


# Experiment name.
model_config_file = os.path.basename(args.model_config_file)
dataset_name, trial_id, _ = model_config_file.split(SEP)
exp_name = SEP.join([dataset_name, trial_id])
exp_name = SEP.join([exp_name, "split_%d" % args.split, "seed_%d" % args.seed])

# Result paths.
result_path = args.result_path or os.path.join(args.path, "results")
if not os.path.exists(result_path):
    os.makedirs(result_path)
if args.save_model or args.fine_tune or args.evaluate:
    model_path = args.model_path or os.path.join(args.path, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
else:
    model_path = None
if args.save_log:
    log_path = args.log_path or os.path.join(args.path, "logs", exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
else:
    log_path = None

trainer = MDNTrainer(
    model=model,
    device=args.device,
    criterions=criterions,
    optimizer=optimizer,
    dataloaders=dataloaders,
    metrics=metrics,
    earlystop_metric_name="survival_loss",
    batch_size=train_config["batch_size"],
    num_epochs=args.num_epochs if not args.evaluate else -1,
    patience=args.patience if not args.evaluate else -1,
    grad_clip=train_config["grad_clip"],
    result_path=result_path,
    model_path=model_path,
    log_path=log_path if not args.evaluate else None,
    log_step=args.log_interval,
    exp_name=exp_name,
    verbose=args.verbose,
    fine_tune=(args.fine_tune or args.evaluate),
    debug=args.debug)


if not args.evaluate:
    trainer.train()
else:
    if hasattr(trainer.model, "set_last_eval"):
        trainer.model.set_last_eval(True)
    trainer.eval(phase="valid")
    valid_metric_value_dict = trainer.get_metric_value_dict(phase="valid")
    trainer.eval(phase="test")
    test_metric_value_dict = trainer.get_metric_value_dict(phase="test")
    eval_path = os.path.join(result_path, "eval")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    result_file = SEP.join(["%s.pkl" % exp_name])
    result_file = os.path.join(eval_path, result_file)
    result = {
        "valid": [valid_metric_value_dict], "test": [test_metric_value_dict]}
    with open(result_file, "wb") as f:
        pickle.dump(result, f)
    trainer.printer.print(result, level=1)

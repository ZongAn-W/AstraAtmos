import random
import numpy as np
import torch

RUN_SEEDS = (11,)


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_seed_metrics(seed_metrics):
    return {}


def print_seed_summary(label, summary):
    print(f"{label} single-seed summary disabled in AstraAtmos")

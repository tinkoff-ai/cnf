import random

import numpy as np
import torch


# noinspection PyUnresolvedReferences
def fix_random_seeds(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

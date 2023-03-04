import logging
from pathlib import Path

import numpy as np
import math


class Sentence:
    aminos = list('AGVLIFWPDEKRHSTNQYCMX')
    token2int = {x: i for i, x in enumerate(list(aminos))}
    int2token = {i: x for i, x in enumerate(list(aminos))}


class Mcts:
    cut = 40
    c_param = 1. / math.sqrt(2.)
    mode = 'full'  # single, full
    best = {'scaffold': '', 'score': 0.}
    acc_cut = 1.
    n_iters = 30000
    max_iter = 30000


class Common:
    data_path = Path('../data/del1/cluster.csv')
    logging_level = logging.INFO
    seed = 42
    pep_len = 12
    fix_len = 6
    variable_len = 6
    optional_len = 0
    n_options = 3
    enrich_cut = 3
    position_scores = np.array(
        [0.00125, 0.001091, 0.000945, 0.000964, 0.001412, 0.001424, 0.000491, 0.00088, 0.000995, 0.001116, 0.000755, 0.00116]
    )  # from ../notebooks/Score.ipynb
    init_scaffold_np = np.array([Sentence.token2int[x] for x in ''])

    np_len = fix_len + optional_len * n_options
    positions = np.argsort(position_scores)

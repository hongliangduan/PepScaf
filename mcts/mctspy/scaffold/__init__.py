import random
from typing import Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from .common import ScafState, AbstractScafAction
from z.utils import get_logger
from conf import Sentence, Mcts, Common
from scaf import Pep, Amino, np2scaffold

log = get_logger(__name__)
log.setLevel(Common.logging_level)

class ScafMove(AbstractScafAction):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Next pep is: {repr(self)}"

    def __repr__(self):
        return Sentence.int2token[self.value]


class ScafGenState(ScafState):

    def __init__(self, scaffold_np: ndarray, data_set, action_set):
        self.scaffold_np = scaffold_np
        self.data_set = data_set
        self.action_set = action_set

    @property
    def result(self):
        # Checking for compeletion
        if len(self.scaffold_np) < Common.np_len:
            return None
        scaffold = np2scaffold(self.scaffold_np)
        score = scaffold(self.data_set)
        Mcts.acc_cut = (Mcts.acc_cut + score)/2  # 移动平均
        if Mcts.acc_cut < 1.:
            Mcts.acc_cut = 1.
        if score > Mcts.best['score']:
            Mcts.best['scaffold'] = repr(scaffold)
            Mcts.best['score'] = score
        if score > Mcts.acc_cut:
            log.debug(f'Score of {repr(scaffold)} is {score:.1f}, which is greater than {Mcts.acc_cut}, now best score is {Mcts.best}')
            return 1
        else:
            return 0

    def is_scaffold_over(self):
        return self.result is not None

    def move(self, move):
        new_pepgen = np.append(np.copy(self.scaffold_np), move.value)
        return type(self)(new_pepgen, self.data_set, self.action_set)

    def __len__(self):
        if len(self.scaffold_np) <= Common.fix_len:
            return len(self.scaffold_np)
        else:
            return (len(self.scaffold_np) - Common.fix_len)//Common.n_options + Common.fix_len
    def get_actions(self):
        action_set = self.action_set[Common.positions[len(self)]].keys()
        return [ScafMove(Sentence.token2int[value]) for value in action_set]

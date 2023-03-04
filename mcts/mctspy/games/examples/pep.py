import random
from typing import Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from mctspy.games.common import PepState, AbstractGenAction
from mcts.utils_tmp import get_logger
from conf import Sentence, Mcts

log = get_logger(__name__)


class PepMove(AbstractGenAction):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Next pep is: {repr(self)}"

    def __repr__(self):
        return Sentence.int2token[self.value]


class PepGenState(PepState):
    def __init__(self, pep: ndarray, predictor, action_set):
        self.pep = pep
        self.predictor = predictor
        # different actions for different positions
        self.action_set = action_set
        self.cut = Mcts.cut

    @property
    def result(self):
        # Checking for compeletion
        if len(self.pep) < 12:
            return None
        score = self.predictor(self.pep)
        if score > self.cut:
            log.info(
                f'Score of {"".join([Sentence.int2token[i] for i in self.pep])} is {score:.1f}, which is greater than {self.cut}'
            )
            return 1
        else:
            return 0

    def is_peptide_over(self):
        return self.result is not None

    def move(self, move):
        new_pepgen = np.append(np.copy(self.pep), move.value)
        return type(self)(new_pepgen, self.predictor, self.action_set)

    def get_actions(self):
        return [PepMove(value) for value in self.action_set[len(self.pep)]]

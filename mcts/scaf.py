import random
from collections import Counter
from typing import List, Union
import numpy as np
from conf import Sentence, Common


class Amino:
    def __init__(self, pos, token, alter=False):
        self.pos = pos
        self.token = token
        self.alter = alter

    @property
    def ids(self):
        return [Sentence.token2int[t] for t in self.token]

    def __str__(self):
        return f'pos: {self.pos}, token: {self.token}'

    def __len__(self):
        return len(self.token)

    def __repr__(self):
        return self.token


class Pep:
    def __init__(self, sentence: str):
        self.sentence = sentence

    @property
    def ids(self):
        return [Sentence.token2int[t] for t in self.sentence]

    @property
    def numpy(self):
        return np.array(self.ids)

    def __str__(self):
        return self.sentence

    def __repr__(self):
        return self.sentence

    def __len__(self):
        return len(self.sentence)


class Scaffold:
    def __init__(self, scaffold_str: str = None, fix: List[Amino] = None, optional: List[Amino] = None,
                 variable: List[Amino] = None):
        self.scaffold_str = None
        if scaffold_str:
            self.scaffold_str = scaffold_str
            self._from_str()
        else:
            self.fix = fix
            self.optional = optional
            self.variable = variable

    def _from_str(self):
        fix = []
        optional = []
        variable = []
        tmp = []
        flag = 1
        pos = 0
        for token in self.scaffold_str:
            if token == 'X' and flag:
                variable.append(Amino(pos, token))
                pos += 1
            elif token in Sentence.aminos and flag:
                fix.append(Amino(pos, token))
                pos += 1
            elif token == '[':
                flag = 0
                continue
            elif token == ']':
                flag = 1
                optional.append(Amino(pos, ''.join(tmp)))
                pos += 1
                tmp = []
                continue
            else:
                if flag == 0:
                    tmp.append(token)
        self.fix = fix
        self.optional = optional
        self.variable = variable

    def numpy(self):
        # check match-ability for position_scores and scaffold
        poses = np.argsort(Common.position_scores)
        fix_poses = [amino.pos for amino in self.fix]
        variable_poses = [amino.pos for amino in self.variable]
        assert set(fix_poses) == set(poses[:Common.fix_len]) and set(variable_poses) == set(
            poses[-Common.variable_len:]), "pos is not match"

        out = []
        for pos in poses[:Common.fix_len]:
            for amino in self.fix:
                if pos == amino.pos:
                    out.append(Sentence.token2int[amino.token])
        # middle position *2
        for pos in poses[Common.fix_len:Common.fix_len + Common.optional_len]:
            for amino in self.optional:
                if pos == amino.pos:  # get its tokens
                    out.extend([Sentence.token2int[t] for t in amino.token])
        return np.array(out)

    def __len__(self):
        return len(self.fix) + len(self.variable) + len(self.optional)

    def predict(self, pep: str):
        _pep = Pep(pep)
        flag = True
        for amino in self.fix:
            if _pep.sentence[amino.pos] != amino.token:
                flag = False
        for amino in self.optional:
            if _pep.sentence[amino.pos] not in amino.token:
                flag = False
        return flag

    def __call__(self, pep: Union[str, List]):
        if isinstance(pep, str):
            return self.predict(pep)
        if isinstance(pep, List):
            return sum([self.predict(p) for p in pep])

    def __str__(self):
        return f'Scaffold: "{repr(self)}" with length of {len(self)}'

    def __repr__(self):
        if self.scaffold_str:
            return self.scaffold_str
        else:
            out = list('X' * len(self))
            for amino in self.fix:
                out[amino.pos] = amino.token
            for amino in self.optional:
                out[amino.pos] = "[" + amino.token + "]"
            return "".join(out)

    def span(self, action_set: List[Counter]) -> List[str]:
        pep_str = ['']*len(self)
        for amino in self.fix:
            pep_str[amino.pos] = amino.token
        for amino in self.variable:
            pep_str[amino.pos] = random.choice(list(action_set[amino.pos].keys()))
        for amino in self.optional:
            pep_str[amino.pos] = random.choice(amino.token)
        return pep_str

def _split_group(a: list, step=Common.n_options):
    return [a[i:i + step] for i in range(0, len(a), step)]


def np2scaffold(arr, fix_len=Common.fix_len, optional_len=Common.optional_len, variable_len=Common.variable_len,
                n_options=Common.n_options):
    positions = np.argsort(Common.position_scores)
    arr = arr.tolist()
    fix_positions = positions[:fix_len]
    variable_positions = positions[-variable_len:]
    optional_positions = positions[fix_len:fix_len + optional_len]
    fix_content = [Sentence.int2token[i] for i in arr[:fix_len]]
    optional_content = [''.join([Sentence.int2token[i] for i in x]) for x in
                        _split_group(arr[-optional_len * n_options:], step=n_options)]

    optional_len = []
    fix = [Amino(pos, token) for pos, token in zip(fix_positions, fix_content)]
    variable = [Amino(pos, 'X') for pos in variable_positions]
    optional = [Amino(pos, token, alter=True) for pos, token in zip(optional_positions, optional_content)]

    return Scaffold(fix=fix, variable=variable, optional=optional)

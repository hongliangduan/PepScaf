import logging

import torch
from torch import nn, softmax

from utils import truncate_tokens_pair, get_logger
import tokenization
import models
import checkpoint
from numpy import ndarray

aminos = list('AGVLIFWPDEKRHSTNQYCM')
token2int = {x: i for i, x in enumerate(list(aminos))}
int2token = {k: x for x, k in token2int.items()}

log = get_logger(__name__)


class Pipeline(object):
    """ Preprocess Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """

    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor  # e.g. text normalization
        self.tokenize = tokenize  # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)

        tokens_a = self.tokenize(self.preprocessor(text_a))

        tokens_b = self.tokenize(self.preprocessor(text_b)) \
            if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """

    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """

    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer  # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)  # token type ids
        input_mask = [1] * (len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class PepEncoder():
    def __init__(self, vocab, max_len=28):
        self.max_len = max_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=False)
        self.pipeline = [
            Tokenizing(self.tokenizer.convert_to_unicode, self.tokenizer.tokenize),
            AddSpecialTokensWithTruncation(self.max_len),
            TokenIndexing(self.tokenizer.convert_tokens_to_ids,
                          ('0', '1'), self.max_len)
        ]

    def __call__(self, label, sentence, next):
        instance = ' '.join(list(label)), ' '.join(list(sentence)), ' '.join(list(next))
        for proc in self.pipeline:
            instance = proc(instance)
        return [torch.tensor(x, dtype=torch.long) for x in instance]


class Classifier(nn.Module):
    """ Classifier with Transformer """

    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h, attn = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits, attn, pooled_h  # we also extract the h for visualization with umap


class Predictor():
    def __init__(self, model, encoder, device):
        self.model = model
        self.encoder = encoder
        self.device = device

    def __call__(self, pep: ndarray):
        pep_str = ''.join([int2token[p] for p in pep])
        # item = ('1', pep_str, pep_str[:3])
        item = ('1', pep_str, '')
        pep = self.encoder(*item)
        data = [x.to(self.device).unsqueeze(0) for x in pep]
        input_ids, segment_ids, input_mask, label_id = data

        logits, attn, pooled_h = self.model(input_ids, segment_ids, input_mask)
        return torch.argmax(softmax(logits, dim=1), dim=1).item()


if __name__ == '__main__':
    ecd = PepEncoder('../data/vocab.txt')
    # print(ecd('1', 'ABMMMBBAAAA', 'ABM'))
    print(ecd('1', 'ABMMMBBAAAA', ''))

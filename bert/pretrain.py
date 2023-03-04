"""Pretrain transformer with Masked LM and Sentence Classification"""
from random import randint, shuffle
from random import random as rand
import fire
import pandas as pd
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import logging
from bert import tokenization, models, optim, train
from bert.utils import set_seeds, get_device, get_random_word, truncate_tokens_pair

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline()  # throw away an incomplete sentence


class PepPairDataLoader:
    """ Load Peptide pair (sequential or random order) from corpus """

    def __init__(
        self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]
    ):
        super().__init__()
        self.df = pd.read_csv(file, index_col=0)
        self.num_data = len(self.df)
        self.tokenize = tokenize  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.i = 0

    def cycle_idx(self, i, col):
        if abs(i) < self.num_data:
            return self.df.at[i, col]
        else:
            return self.df.at[i % self.num_data, col]

    def __iter__(self):  # iterator to load data
        while True:
            batch = []
            for j in range(self.batch_size):
                # print(f'iloc: {self.i}')
                is_next = rand() < 0.5
                tokens_a = list(self.cycle_idx(self.i, "seq"))
                if is_next:
                    tokens_b = list(self.cycle_idx(self.i, "next"))
                else:
                    tokens_b = list(
                        self.cycle_idx(randint(0, self.num_data - 1), "next")
                    )
                instance = (is_next, tokens_a, tokens_b)

                for proc in self.pipeline:
                    instance = proc(instance)

                batch.append(instance)
                self.i += 1

            # To Tensor
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline:
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [
            i for i, token in enumerate(tokens) if token != "[CLS]" and token != "[SEP]"
        ]
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8:  # 80%
                tokens[pos] = "[MASK]"
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        return (
            input_ids,
            segment_ids,
            input_mask,
            masked_ids,
            masked_pos,
            masked_weights,
            is_next,
        )


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"

    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        # input_ids=(b, l), segment_id=(b, l), input_mask=(b, l), masked_pos=(b, n)
        h, _ = self.transformer(input_ids, segment_ids, input_mask)  # (b, l, d)
        pooled_h = self.activ1(self.fc(h[:, 0]))  # (b, l)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # (b, n, d)
        h_masked = torch.gather(h, 1, masked_pos)  # (b, n, d)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))  # (b, n, d)
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # (b, n, v)
        logits_clsf = self.classifier(pooled_h)  # (b, 2)

        return logits_lm, logits_clsf


def main(
    train_cfg="config/pretrain.json",
    model_cfg="config/bert.json",
    data_file="data/FSA/pretrain.csv",
    model_file=None,
    data_parallel=True,
    vocab="data/vocab.txt",
    save_dir="models",
    log_dir="logs",
    max_len=28,
    max_pred=8,
    mask_prob=0.15,
):
    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab, do_lower_case=False
    )  # 注意这里要 False
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [
        Preprocess4Pretrain(
            max_pred,
            mask_prob,
            list(tokenizer.vocab.keys()),
            tokenizer.convert_tokens_to_ids,
            max_len,
        )
    ]
    data_iter = PepPairDataLoader(
        data_file, cfg.batch_size, tokenize, max_len, pipeline=pipeline
    )

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction="none")
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir)  # for tensorboardX

    def get_loss(model, batch, global_step):  # make sure loss is tensor
        (
            input_ids,
            segment_ids,
            input_mask,
            masked_ids,
            masked_pos,
            masked_weights,
            is_next,
        ) = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
        loss_lm = (loss_lm * masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next)  # for sentence classification
        writer.add_scalars(
            "data/scalar_group",
            {
                "loss_lm": loss_lm.item(),
                "loss_clsf": loss_clsf.item(),
                "loss_total": (loss_lm + loss_clsf).item(),
                "lr": optimizer.get_lr()[0],
            },
            global_step,
        )
        return loss_lm + loss_clsf

    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == "__main__":
    fire.Fire(main)

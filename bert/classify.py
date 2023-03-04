import pickle
import itertools
import csv
from pathlib import Path
import fire
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import auroc
from bert import tokenization, optim, train, models
from bert.utils import (
    set_seeds,
    get_device,
    truncate_tokens_pair,
    iter_count,
    get_logger,
    pbar,
)
import logging
from torch.nn.functional import softmax

log = get_logger(__name__)
log.setLevel(logging.INFO)


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """

    labels = None

    def __init__(self, file, pipeline=[]):  # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter=",", quotechar=None)

            @pbar(
                self.get_instances(lines),
                totol=iter_count(file),
                description="Loading data",
            )
            def do(i, pipeline, data):
                for proc in pipeline:  # a bunch of pre-processing
                    i = proc(i)
                data.append(i)

            do(pipeline=pipeline, data=data)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class PepCLS(CsvDataset):
    """ Dataset class for MRPC"""

    labels = ("0", "1")  # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):  # skip the header
            # label, text (0 for index)
            yield " ".join(list(line[2])), " ".join(list(line[1])), None


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {"pep_cls": PepCLS}
    return table[task]


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
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b else []

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
        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_b = tokens_b + ["[SEP]"] if tokens_b else []

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
        return (
            logits,
            attn,
            pooled_h,
        )  # we also extract the h for visualization with umap


def main(
    task="pep_cls",
    train_cfg="config/train.json",
    model_cfg="config/bert.json",
    data_file=None,
    model_file=None,
    data_parallel=True,
    pretrain_file=None,
    mode="train",
    vocab="data/vocab.txt",
    save_dir=None,
    max_len=14,
):
    log.info("loading config...")
    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    set_seeds(cfg.seed)
    log.info(f"Loading Data from {data_file}...")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=False)
    # task dataset class according to the task
    TaskDataset = dataset_class(task)
    pipeline = [
        Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        AddSpecialTokensWithTruncation(max_len),
        TokenIndexing(tokenizer.convert_tokens_to_ids, TaskDataset.labels, max_len),
    ]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    log.info(f"loading model from {model_file}...")
    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()
    log.info("start training...")
    trainer = train.Trainer(
        cfg, model, data_iter, optim.optim4GPU(cfg, model), save_dir, get_device()
    )

    if mode == "train":

        def get_loss(model, batch, global_step):  # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits, _, _ = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == "eval":

        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits, attns, pooled_h = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()  # .cpu().numpy()
            accuracy = result.mean()
            # 扩充一维用于 cat，cat 的结果是 12 layers
            # log.debug(f'in a batch evaluate, attns: {attns[0].shape}')
            # batch, layers, head, long, long
            rt_attns = torch.cat([t.unsqueeze(1) for t in attns], dim=1)
            # log.debug(f'in a batch evaluate, rt_attns: {rt_attns.shape}')
            return (
                accuracy,
                result,
                softmax(logits, dim=1),
                label_id,
                rt_attns,
                pooled_h,
            )

        results, logits, labels, attns, pooled_hs = trainer.eval(
            evaluate, model_file, data_parallel
        )
        total_accuracy = torch.cat(results).mean().item()
        auc_score = auroc(torch.cat(logits), torch.cat(labels), num_classes=2)
        log.info(f"Accuracy: {total_accuracy}")
        log.info(f"AUC Score: {auc_score}")

        if save_dir:
            log.info("Output Attention Matrix")
            rlines = []
            for data in dataset:
                phla = data[0]
                rdict = {v: k for k, v in tokenizer.vocab.items()}
                line = []
                for id in phla.tolist():
                    line.append(rdict[id])
                rlines.append(line)
            log.debug(rlines[0])
            SAVE_DIR = Path(save_dir)
            SAVE_DIR.mkdir(exist_ok=True, parents=True)
            with open(SAVE_DIR / "attns.pkl", "wb") as fattns:
                # 这里拼接一下得到一整个张量
                # num, layes, head, len, len
                pickle.dump(torch.cat(attns), fattns)
            with open(SAVE_DIR / "tokens.pkl", "wb") as ftokens:
                pickle.dump(rlines, ftokens)
            with open(SAVE_DIR / "pooled_hs.pkl", "wb") as fpooled_hs:
                pickle.dump(torch.cat(pooled_hs), fpooled_hs)
            with open(SAVE_DIR / "labels.pkl", "wb") as flabels:
                pickle.dump(torch.cat(labels), flabels)
            with open(SAVE_DIR / "logits.pkl", "wb") as flogits:
                pickle.dump(torch.cat(logits), flogits)


if __name__ == "__main__":
    fire.Fire(main)

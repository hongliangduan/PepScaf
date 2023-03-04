from collections import Counter
from utils import get_logger, set_seeds, get_device
import pandas as pd
import numpy as np
from pathlib import Path
from components import PepEncoder, Classifier, Predictor
import models, torch
import torch
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from conf import Common, Mcts, Sentence
from scaf import Scaffold, np2scaffold, Pep
from mctspy.scaffold import ScafGenState
from mctspy.tree.nodes import ScafMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch


log = get_logger(__name__)
log.setLevel(Common.logging_level)

seed = Common.seed
set_seeds(seed)
log.info(f"Set seeds: {seed}")
DATA_PATH = Common.data_path
enrich_cut = Common.enrich_cut  # use data with enrich > 10
pep_len = Common.pep_len
optional_len = Common.optional_len
n_options = Common.n_options
variable_len = Common.variable_len
fix_len = Common.fix_len

log.info(f"Loading data from {DATA_PATH}")
_df = pd.read_csv(DATA_PATH)
log.info(f"Delete data with len unequal to {pep_len}")

# make sure the length of peptide is 12
_df = _df[_df["seq"].apply(len) == pep_len]
if enrich_cut:
    df = _df[_df["enrichment"] > enrich_cut]
data = df["seq"].tolist()
all_data = _df["seq"].tolist()
n = len(data)
all_n = len(all_data)
for i, x in enumerate(data):
    assert len(x) == pep_len, f"No.{i}'s len is not {pep_len}"
log.info(f"Number of data with enrich > {enrich_cut} and len = {pep_len} is {n}")

log.info(f"Get Action Set from Data Set")
action_set = []


def ext(pep: str, i: int):
    return pep[i]


for i in range(Common.pep_len):
    action_set.append(Counter(_df["seq"].apply(ext, args=(i,))))
log.debug(action_set)


log.info("Run MCTS...")
log.info(f"Init scaffold_np is: {Common.init_scaffold_np}")
log.info(f"Positions: {np.argsort(Common.position_scores)}")
pieces = {1: "Positive", 0: "Negative"}

initial_scaf_state = ScafGenState(
    scaffold_np=Common.init_scaffold_np, data_set=data, action_set=action_set
)


def scaffold_generator(state, n_iters):
    root = ScafMonteCarloTreeSearchNode(state=state)
    mcts = MonteCarloTreeSearch(root)
    best_nodes = mcts.best_actions(n_iters)
    return best_nodes


if Mcts.mode == "single":
    if initial_scaf_state.result is None:
        best_nodes = scaffold_generator(initial_scaf_state, Mcts.n_iters)
        for node, score in best_nodes:
            log.info(
                f"Next is {Sentence.int2token[node.state.scaffold_np[-1]]}'s score: {score}"
            )
    else:
        log.info(f"Scaffold Result is {initial_scaf_state.result}")
        final_scaffold = np2scaffold(initial_scaf_state.scaffold_np)
        log.info(f"Final Scaffold is {final_scaffold}")
        n_active = final_scaffold(data)
        all_n_active = final_scaffold(all_data)
        log.info(
            f"When applied to data with {n} items whose enrich is above {enrich_cut}: {n_active}({100 * n_active / n:.2f}%) satisfied"
        )
        log.info(
            f"When applied to whole data with {all_n} items: {all_n_active}({100 * all_n_active / all_n:.2f}%) satisfied"
        )
        log.info(
            f"Improvement rate is {100 * n_active * all_n / n / all_n_active:.2f}%"
        )


def second(scaffold: Scaffold):
    """Predict Scaffold Spaned peptide using BERT trained on L1-DEL"""
    DATA_DIR = Path("../data")
    CONFIG_DIR = Path("../bert/config")
    MODEL_DIR = Path("../bert/exp/train")
    VOCAB = DATA_DIR / "vocab.txt"
    MODEL_CFG = CONFIG_DIR / "bert.json"
    MODEL = MODEL_DIR / "model_steps_19000.pt"
    log.info(f"Loading Config from {MODEL_CFG}...")
    model_cfg = models.Config.from_json(MODEL_CFG)
    device = get_device()
    encoder = PepEncoder(VOCAB, max_len=14)
    log.info(f"Init Model...")
    model = Classifier(model_cfg, 2)  # Binary Classification
    log.info(f"Set Eval Mode...")
    model.eval()
    log.info(f"Loading Model from {MODEL}...")
    model.load_state_dict(torch.load(MODEL))
    log.info(f"Model to Device...")
    model.to(device)
    predictor = Predictor(model, encoder, device)
    n = 1000  # Spaning Space
    pep_list = ["".join(scaffold.span(action_set=action_set)) for i in range(n)]
    pep_set = set(pep_list)
    log.info(f"data num is {len(pep_set)}")
    score = 0.0
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        "Elapsed:",
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        for pep in progress.track(pep_set, description="Predicting..."):
            score += predictor(Pep(pep).numpy)
    log.info(f"{100*score/len(pep_set):.2f}% peptides was pedicted into positive.")


# if Mcts.mode == 'full':
#     state = initial_scaf_state
#     while state.result is None:
#         rest = Common.np_len - len(state.scaffold_np)
#         n_iters = 5 * 10 ** rest
#         max_iter = Mcts.max_iter if Mcts.max_iter else Mcts.n_iters
#         if n_iters >= max_iter:
#             n_iters = max_iter
#         log.info(f'Searching space is {n_iters}')
#         best_nodes = scaffold_generator(state, n_iters)
#         best_node = best_nodes[0][0]
#         score = best_nodes[0][1]
#         candidate_node = best_nodes[1][0]
#         candidate_score = best_nodes[1][1]
#         log.info(
#             f"Next is {Sentence.int2token[best_node.state.scaffold_np[-1]]}'s score: {score:.4f}.")
#         log.info(
#             f"Candidate is {Sentence.int2token[candidate_node.state.scaffold_np[-1]]}'s score: {candidate_score:.4f}.")
#         state = best_node.state

#     log.info(f'Scaffold Result is {state.result}')
#     final_scaffold = np2scaffold(state.scaffold_np)
#     log.info(f'Extracted Scaffold is {final_scaffold}')
#     n_active = final_scaffold(data)
#     all_n_active = final_scaffold(all_data)
#     log.info(
#         f'\n'
#         f'When Applied to Data with {n} Items Whose Enrich is Above {enrich_cut}: {n_active}({100 * n_active / n:.2f}%) Aatisfied.\n'
#         f'When Applied to All the Data with {all_n} Items: {all_n_active}({100 * all_n_active / all_n:.2f}%) Aatisfied.\n'
#         f'Improvement Rate is {100 * n_active * all_n / n / all_n_active:.2f}%.'
#     )
# second(final_scaffold)
second(Scaffold("XXYYXXLYGXLX"))

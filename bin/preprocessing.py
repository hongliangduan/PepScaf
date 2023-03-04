import pandas as pd
from pathlib import Path
from z.utils import show_ratio, set_seeds, split_df, get_logger
from z.pep import pep_gen

# Config
DIR = Path("data/del1")
RAW = DIR / "raw.csv"
L1 = DIR / "data.csv"
BALANCED = DIR / "balanced.csv"
TRAIN = DIR / "train.csv"
TEST = DIR / "test.csv"
pep_len = 12  # Remain peptides of 12 length
seed = 42

DIR.mkdir(exist_ok=True)
set_seeds(seed)  # Set random seed
log = get_logger(level="INFO")

# Clean Data
df = pd.read_csv(RAW, index_col=0)
log.info(f"Number of valid data: {len(df)}")
df["seq_len"] = df.seq.apply(len)
df = df[df["seq_len"] == pep_len]  # Delete Non-12-length peptides
df.drop(columns=["seq_len"], inplace=True)

# Balance Data
n_negatives = len(df)
pep_gens = pep_gen(n_negatives)  # Generate equal quality of negative random peptides
pep_gens_df = pd.DataFrame(
    {"enrichment": [0] * n_negatives, "seq": pep_gens, "label": [0] * n_negatives}
)
df["label"] = 1
df_all = pd.concat([df, pep_gens_df])  # Merge Positive and Negative Peptides.
df_all.drop_duplicates(inplace=True, keep="first")  # Keep the origin peptide
df_all.reset_index(drop=True, inplace=True)  # Fix the index


# df_all['next'] = df_all.seq.apply(lambda x: x[:3])  # Adapt the MLM style training, no special meanings
# balanced_df = pd.DataFrame(
#     {'seq': df_all.seq, 'enrichment': df_all.enrichment, 'label': df_all.label, 'next': df_all.next})

balanced_df = pd.DataFrame({"seq": df_all.seq, "label": df_all.label})

train_df, test_df = split_df(
    balanced_df, shuf=True, val=False, random_state=seed
)  # 8:2

log.info(f"Save L1-DEL in lentgh of 12 to {L1}")
df.reset_index(drop=True).to_csv(L1)
balanced_df.to_csv(BALANCED)  # Pretrain dataset
train_df.reset_index(drop=True).to_csv(TRAIN)  # Train set
test_df.reset_index(drop=True).to_csv(TEST)  # Test set

log.info("Positive/Negative Ratio:")
log.info("Pretrain Dataset:")
show_ratio(balanced_df)
log.info("Train Dataset:")
show_ratio(train_df)
log.info("Test Dataset:")
show_ratio(test_df)

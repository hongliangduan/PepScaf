import pandas as pd
from Bio import SeqIO
from pathlib import Path
from z.utils import get_logger, set_seeds
from z.pep import record_iters, get_cluster, cluster2df, pep_gen
import subprocess

# Config
DIR = Path('data/del1')
CACHE = DIR / '.cache'
DATA = DIR / 'data.csv'
FASTA = CACHE / 'data.faa'
DATA_OUT = CACHE / 'data.out'
CLUSTER = CACHE / 'data.out.clstr'
CSV = DIR / 'cluster.csv'
BALCANCED_CSV = DIR / 'cluster_balanced.csv'
NO_CLUSTER = 2  # Target Cluster
SEED = 42

# Program goes
set_seeds(SEED)
DIR.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
log = get_logger(__name__)

log.info('Convert *.df to *.fasta')
df = pd.read_csv(DATA, index_col=0)
log.info(f'Number of positive samples are {len(df)}.')
log.info(f'{FASTA} Generated')  # Write into fasta
SeqIO.write(record_iters(df['seq'], df.index,
            df['enrichment']), FASTA, "fasta")

log.info('Run CD-HIT...')
cmd = f'./bin/cd-hit -i {FASTA} -o {DATA_OUT} -c 0.8 -n 5 -M 16000 -d 0 -T 8'
p = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, shell=False)
p.wait()

log.info('Get Target Cluster')
target_cluster = get_cluster(CLUSTER, no_cluster=NO_CLUSTER)
log.info(f'Size of Target Cluster: {len(target_cluster)}')
df_target_cluster = cluster2df(target_cluster, refer_df=df)
pd.DataFrame(
    {'enrichment': df_target_cluster['enrichment'], 'seq': df_target_cluster['seq'], 'label': df_target_cluster['label']}).to_csv(CSV)
log.info(f'Target cluster was written into {CSV}'
)

log.info('Get Balanced Target Cluster')
n_negs = len(df_target_cluster)
neg_peps = pep_gen(n_negs)
neg_peps_df = pd.DataFrame({'seq': neg_peps, 'enrichment': [
                           0] * n_negs,  'label': [0] * n_negs})
neg_peps_df['next'] = neg_peps_df['seq'].apply(lambda x: x[:3])
pos_peps_df = pd.DataFrame(
    {'seq': df_target_cluster['seq'],
     'enrichment': df_target_cluster['enrichment'],
     'label': df_target_cluster['label']})
pos_peps_df['next'] = pos_peps_df['seq'].apply(lambda x: x[:3])
df_balanced_target_cluster = pd.concat(
    [pos_peps_df, neg_peps_df]).reset_index(drop=True)
# df_balanced_target_cluster.to_csv(BALCANCED_CSV)
pd.DataFrame({'seq': df_balanced_target_cluster['seq'], 'label': df_balanced_target_cluster['label']}).to_csv(BALCANCED_CSV)  # for non-pretrain
log.info(f'Balanced Target Cluster was Written into {BALCANCED_CSV}')

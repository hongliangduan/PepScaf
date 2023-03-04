CUDA_LAUNCH_BLOCKING=1
pushd bert
python classify.py \
  --task='pep_cls' \
  --train_cfg='config/train.json' \
  --data_file='../data/del1/cluster_balanced.csv' \
  --model_cfg='config/bert.json' \
  --mode='eval' \
  --model_file='exp/train/model_steps_19000.pt' \
  --vocab='../data/vocab.txt' \
  --save_dir='exp/attns/cluster' \
  --max_len=14
popd
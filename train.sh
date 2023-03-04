CUDA_LAUNCH_BLOCKING=1
pushd bert
python classify.py \
  --task='pep_cls' \
  --train_cfg='config/train.json' \
  --data_file='../data/del1/train.csv' \
  --model_cfg='config/bert.json' \
  --mode='train' \
  --vocab='../data/vocab.txt' \
  --save_dir='exp/train' \
  --max_len=14
popd


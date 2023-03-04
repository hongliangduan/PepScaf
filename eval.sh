CUDA_LAUNCH_BLOCKING=1
pushd bert
python classify.py \
  --task='pep_cls' \
  --train_cfg='config/train.json' \
  --data_file='../data/del1/test.csv' \
  --model_cfg='config/bert.json' \
  --mode='eval' \
  --model_file='exp/train/model_steps_19000.pt' \
  --vocab='../data/vocab.txt' \
  --save_dir='exp/attns/test' \
  --max_len=14
popd
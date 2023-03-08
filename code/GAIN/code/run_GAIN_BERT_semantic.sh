#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_BERT_base Training Shell Script--------------------

if true; then
  model_name=GAIN_BERT_base
  lr=0.001
  batch_size=10
  test_batch_size=16
  epoch=300
  test_epoch=5
  log_step=20
  save_model_freq=3
  negativa_alpha=4

  CUDA_VISIBLE_DEVICES=6 python3 train.py \
    --train_set ../data/train_annotated.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT.pkl \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --log_step ${log_step} \
    --save_model_freq ${save_model_freq} \
    --negativa_alpha ${negativa_alpha} \
    --gcn_dim 916 \
    --gcn_layers 2 \
    --bert_hid_size 768 \
    --bert_path bert-base-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --coslr \
    --use_semantic_entity_embedding \
    # >logs/train_${model_name}_ours.log 2>&1
fi
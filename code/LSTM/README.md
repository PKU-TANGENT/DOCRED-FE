# Baseline code
The code here is mainly from [DOCRED](https://github.com/thunlp/DocRED/tree/master/code), we reuse most part of their code and rewrite in some place to fit our baseline.

## Requirements and Installation
python3

pytorch>=1.0

```
pip3 install -r requirements.txt
```

## preprocessing data
Put the downloaded data into data folder and run the following code.


```
python3 gen_data.py --in_path ./data --out_path prepro_data
```

## relation extration

### training:
#### Default

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

#### Double-level entity information

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --use_f_entity_type
```

#### Semantic representation

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --use_semantic_entity_embedding
```

### testing (--test_prefix dev_dev for dev set, dev_test for test set):
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601
```


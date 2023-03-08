# Double Graph Based Reasoning for Document-level Relation Extraction

The code here is mainly from [GAIN](https://github.com/DreamInvoker/GAIN), we reuse most part of their code and rewrite in some place to fit our baseline.

## Package Description
```
GAIN/
├─ code/
    ├── checkpoint/: save model checkpoints
    ├── fig_result/: plot AUC curves
    ├── logs/: save training / evaluation logs
    ├── models/:
        ├── GAIN.py: GAIN model for BERT version
    ├── config.py: process command arguments
    ├── data.py: define Datasets / Dataloader for GAIN-BERT
    ├── test.py: evaluation code
    ├── train.py: training code
    ├── utils.py: some tools for training / evaluation
    ├── *.sh: training / evaluation shell scripts
├─ data/: raw data and preprocessed data about DocRED dataset
    ├── prepro_data/
├─ PLM/
├─ test_result_jsons/: save test result jsons
├─ README.md
```

## Environments

- python         (3.7.4)
- cuda           (10.2)
- Ubuntu-18.0.4  (4.15.0-65-generic)

## Dependencies

- numpy          (1.19.2)
- matplotlib     (3.3.2)
- torch          (1.6.0)
- transformers   (3.1.0)
- dgl-cu102      (0.4.3)
- scikit-learn   (0.23.2)

PS: dgl >= 0.5 is not compatible with our code, we will fix this compatibility problem in the future.

## Preparation

### Dataset
- Download meta data from [Google Drive link](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) shared by DocRED authors and DOCRED-FE data from [Google Drive link](https://drive.google.com/drive/folders/1KYKJBEsD0gZVjAvHSfsIer8-XWeSyJ7m)

- Put `train_annotated.json`, `dev.json`, `test.json`, `word2id.json`, `ner2id.json`, `rel2id.json`, `vec.npy` into the directory `data/`

### (Optional) Pre-trained Language Models
Following the hint in this [link](http://viewsetting.xyz/2019/10/17/pytorch_transformers/?nsukey=v0sWRSl5BbNLDI3eWyUvd1HlPVJiEOiV%2Fk8adAy5VryF9JNLUt1TidZkzaDANBUG6yb6ZGywa9Qa7qiP3KssXrGXeNC1S21IyT6HZq6%2BZ71K1ADF1jKBTGkgRHaarcXIA5%2B1cUq%2BdM%2FhoJVzgDoM7lcmJg9%2Be6NarwsZzpwAbAwjHTLv5b2uQzsSrYwJEdPl7q9O70SmzCJ1VF511vwxKA%3D%3D), download possible required files (`pytorch_model.bin`, `config.json`, `vocab.txt`, etc.) into the directory `PLM/bert-????-uncased` such as `PLM/bert-base-uncased`.

## Training

### Default
```bash
>> cd code
>> ./run_GAIN_BERT_default.sh
```

#### Double-level entity information
```bash
>> cd code
>> ./run_GAIN_BERT_double.sh
```

#### Semantic representation
```bash
>> cd code
>> ./run_GAIN_BERT_semantic.sh
```

## Evaluation

```bash
>> cd code
>> ./eval_GAIN_BERT_xxx.sh
```
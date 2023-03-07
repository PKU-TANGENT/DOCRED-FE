# JEREX: "Joint Entity-Level Relation Extractor"
The code here is mainly from [JEREX](https://github.com/lavis-nlp/jerex), we reuse most part of their code and rewrite in some place to fit our baseline.

## Setup
### Requirements
- Required
  - Python 3.7+
  - PyTorch (tested with version 1.8.1 - see [here](https://pytorch.org/get-started/locally/) on how to install the correct version)
  - PyTorch Lightning (tested with version 1.2.7)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.5.1)
  - hydra-core (tested with version 1.0.6)
  - scikit-learn (tested with version 0.21.3)
  - tqdm (tested with version 4.43.0)
  - numpy (tested with version 1.18.1)
  - jinja2 (tested with version 2.11.3)

### Fetch data
(1) Put the downloaded data in folder data/datasets/docred-fe 

(2) Fetch model checkpoints (joint multi-instance model (end-to-end split) and relation classification multi-instance model (original split)):
```
bash ./scripts/fetch_models.sh
```

## Examples

### End-to-end (joint) model (For Joint Entity and Relation Extraction task)
(1) Train JEREX (joint model) using the end-to-end split:
```
python ./jerex_train.py --config-path configs/docred_joint
```

(2) Evaluate JEREX (joint model) on the end-to-end split (you need to fetch the model first):
```
python ./jerex_test.py --config-path configs/docred_joint
```

### Relation Extraction (only) model (For Relation Classification task)
To run these examples, first download the original DocRED-FE dataset into './data/datasets/docred-fe/' following the instructions of DocRED (see 'https://github.com/thunlp/DocRED' for instructions)

(1) Train JEREX (multi-instance relation classification component) using the DocRED-FE dataset.

#### Default
```
python ./jerex_train.py --config-path configs/docred 
```

#### Double-level entity information
```
python ./jerex_train.py --config-path configs/docred use_f_entity_type=true
```

#### Semantic representation
```
python ./jerex_train.py --config-path configs/docred use_semantic_entity_embedding=true
```

(2) Evaluate JEREX (multi-instance relation classification component) on the original DocRED-FE test set (you need to fetch the model first):
```
python ./jerex_test.py --config-path configs/docred
```

## Configuration / Hyperparameters
- The hyperparameters used in our paper are set as default. You can adjust hyperparameters and other configuration settings in the 'train.yaml' and 'test.yaml' under ./configs
- Settings can also be overriden via command line, e.g.:
```
python ./jerex_train.py training.max_epochs=40
```
- A brief explanation of available configuration settings can be found in './configs.py'
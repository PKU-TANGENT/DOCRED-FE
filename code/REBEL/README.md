# REBEL: Relation Extraction By End-to-end Language generation

The code here is mainly from [rebel](https://github.com/Babelscape/rebel), we reuse most part of their code and rewrite in some place to fit our baseline.

```
Repo structure
| conf  # contains Hydra config files
  | data
  | model
  | train
  root.yaml  # hydra root config file
| data  # data
| datasets  # datasets scripts
| model # model files should be stored here
| src
  | pl_data_modules.py  # LightinigDataModule
  | pl_modules.py  # LightningModule
  | train.py  # main script for training the network
  | test.py  # main script for training the network
| README.md
| requirements.txt
| setup.sh # environment setup script 
```

## Initialize environment
In order to set up the python interpreter we utilize [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
, the script setup.sh creates a conda environment and install pytorch
and the dependencies in "requirements.txt". 

## REBEL Model and Dataset

Model can be downloaded here:

https://osf.io/4x3r9/?view_only=87e7af84c0564bd1b3eadff23e4b7e54

Or you can directly use the model from Huggingface repo:

https://huggingface.co/Babelscape/rebel-large



## Training and testing

There are conf files to train and test each model.:
```
python train.py model=rebel_model data=docred_data train=docred_train
```
Once the model is trained, the checkpoint can be evaluated by running:
```
python test.py model=rebel_model data=docred_data train=docred_train do_predict=True checkpoint_path="path_to_checkpoint"
```
src/model_saving.py can be used to convert a pytorch lightning checkpoint into the hf transformers format for model and tokenizer.


## Datasets

Put the downloaded codes under data/ and rename following the pattern in yaml file.

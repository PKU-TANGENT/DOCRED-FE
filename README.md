# DocRED-FE
Dataset and code for baselines for "DOCRED-FE: A DOCUMENT-LEVEL FINE-GRAINED ENTITY AND RELATION EXTRACTION DATASET"[IEEE eXpress](https://ieeexplore.ieee.org/document/10095786), [arXiv](https://arxiv.org/abs/2303.11141)

Joint entity and relation extraction (JERE) is one of the most
important tasks in information extraction. However, most
existing works focus on sentence-level coarse-grained JERE,
which have limitations in real-world scenarios. In this pa-
per, we construct a large-scale document-level fine-grained
JERE dataset DocRED-FE, which improves DocRED with
Fine-Grained Entity Type. Specifically, we redesign a hierar-
chical entity type schema including 11 coarse-grained types
and 119 fine-grained types, and then re-annotate DocRED
manually according to this schema. Through comprehensive
experiments we find that:

+ DocRED-FE is challenging to existing JERE model.
+ Our fine-grained entity types promote relation classification.

## Cite
If you use the dataset or the code, please cite this paper:
```
@INPROCEEDINGS{10095786,
  author={Wang, Hongbo and Xiong, Weimin and Song, Yifan and Zhu, Dawei and Xia, Yu and Li, Sujian},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={DocRED-FE: A Document-Level Fine-Grained Entity and Relation Extraction Dataset}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095786}}

```

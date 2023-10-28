# PatchView

## Intro
Common Vulnerabilities and Exposures (CVE) is a system for cataloging known security vulnerabilities in publicly released software or firmware. While it is standard procedure for developers to announce a CVE record post the identification and rectification of a software vulnerability, some developers choose to omit this step and merely update their repositories, keeping the vulnerabilities undisclosed. Such actions leave users uninformed and potentially at risk. To this end, we present PatchView, an innovative multi-modal system tailored for the classification of commits as security patches. The system draws upon three unique data modalities associated with a commit: 
1) Time-series representation of developer behavioral data within the Git repository,
2) Commit text messages, 
3) The code content. 

PatchView merges three single-modality sub-models, each adept at interpreting data from its designated source. A distinguishing feature of this solution is its ability to elucidate its predictions by examining the outputs of each sub-model, underscoring its interpretability. Notably, this research pioneers a language-agnostic methodology for security patch classification. Our evaluations indicate that the proposed solution can reveal concealed security patches with an accuracy of 94.52% and F-Score of 95.12%.


![PatchView Design](results/design3.drawio.png "a title")

## Folders
* Data: contains the raw datasets and script files for processing the datasets.
* Models: contains the models used in the repository.
* Results: contains the scripts used to gather the results from W&B and create figures for them.
* Sweeps: contains sweep configuartions for W&B.

## Requirements
* python 3.10
* Pytorch 1.13


## Usage
### Commit Message Model Train
```bash
python main.py --epochs 10 --batch_size 16 --source_model Message --message_model_type roberta --learning_rate 1e-5 --recreate_cache
```

### Behavioural Model Train
```bash
python main.py --activation=tanh  --batch_size=512 --dropout=0.3 --epochs=600 --event_l1=83 --event_l2=41 --event_l3=83 --event_l4=80 --event_window_size=41 --folds=10 --learning_rate=0.0001 --run_fold=7 --source_model=Events
```

### Code Model Train
```bash
python main.py --epochs 100 --learning_rate 1e-5 --dropout 0.8  --recreate-cache --folds 10  --source_model Code  --model_type roberta
```

### Multi-Modality Model Train
```bash
python main.py --epochs 10 --eval_batch_size 16 --train_batch_size 16 -lr 1e-5 --dropout 0.7  --recreate_cache --code_merge_file --source_model Multi
```

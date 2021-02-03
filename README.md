# Semeval 2020, Task 11

## Overview
This repository provides code for the SemEval-2020 Task 11 competition (Detection of Propaganda Techniques in News Articles).

The competition webpage: https://propaganda.qcri.org/semeval2020-task11/

The description of the architecture of models can be found in our paper [Aschern at SemEval-2020 Task 11: It Takes Three to Tango: RoBERTa, CRF, and Transfer Learning](https://www.aclweb.org/anthology/2020.semeval-1.191/).

## Requirements
```
pip install -r ./requirements.txt
```

## Project structure

- `configs`: yaml configs for the system
- `datasets`: contains the task datasets, which can be downloaded from the team competition webpage
- `results`: the folder for submissions
- `span_identification`: code for the task SI
  - `ner`: pytorch-transformers RoBERTa model with CRF (end-to-end)
  - `dataset`: the scripts for loading and preprocessing source dataset
  - `submission`: the scripts for obtaining and evaluating results
- `technique_classification`: code for the task TC (the folder has the same structure as `span_identification`)
- `tools`: tools provided by the competition organizers; contain useful functions for reading datasets and evaluating submissions
- `visualization_example`: example of visualization of results for both tasks

## Running the models

All commands are run from the root directory of the repository.

### Span Identification

1. Configure `configs/si_config.yml` file, if it is needed. data_dir is the path to the cache of original train/eval sub-datasets and their BIO versions. In addition to using the config, it is also possible to specify arguments through the command line.

2. Split the dataset for local evaluation (if `--overwrite_cache`, previous files will be replaced). It will produce files with the BIO-format tagging for spans (B-PROP, I-PROP, O) in your `--data_dir`.
    ```bash
    python -m span_identification --config configs/si_config.yml --split_dataset --overwrite_cache
    ```
3. Train and eval model (the model parameters are specified in the config, you need to change the paths). The use of CRF is regulated by the flag `--use_crf`. For the first run you can use `--model_name_or_path roberta-large`.
    ```bash
    python -m span_identification --config configs/si_config.yml --do_train --do_eval
    ```
4. Apply the trained model to the `test_file` (in BIO-format) specified in the config. It will be created based on the `test_data_folder` folder in case of missing or if the flag `--overwrite_cache` is specified.
    ```bash
    python -m span_identification --config configs/si_config.yml --do_predict
    ```
5. Create the submission file `output_file` in the `result` folder. It will obtain spans from the result files with the token labeling specified in `predicted_labels_files`. At the aggregation stage, the span prediction results are simply joined.
    ```bash
    python -m span_identification --config configs/si_config.yml --create_submission_file
    ```
6. In case you have the correct markup in the `test_file` or gold `--gold_annot_file` (source competition format), you can run the evaluation competition script.
    ```bash
    python -m span_identification --config configs/si_config.yml --do_eval_spans
    ```
7. Use `visualization_example/visualization.ipynb` if you want to visualize labels.

### Technique Classification

Here you need almost the same commands and settings as in the SI task.

1. Configure `configs/tc_config.yml` file, if it is needed.

2. Split the dataset for local evaluation.
    ```bash
    python -m technique_classification --config configs/tc_config.yml --split_dataset --overwrite_cache
    ```
3. Train and eval model. We used two setups with and without flags `--join_embeddings --use_length` (to get our RoBERTa-Joined). For the first run you can use `--model_name_or_path roberta-large`.
    ```bash
    python -m technique_classification --config configs/tc_config.yml --do_train --do_eval
    ```
    or distributed
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 technique_classification --config configs/tc_config.yml --do_train --do_eval
    ```
4. Apply the trained model to the `test_file` specified in the config. It will be created based on the `test_data_folder` folder and `test_template_labels_path` file in case of missing or if the flag `--overwrite_cache` is specified.
    ```bash
    python -m technique_classification --config configs/tc_config.yml --do_predict --join_embeddings --use_length
    ```
5. Create the submission file `output_file`. It will combine predictions from the list `predicted_logits_files` with coefficients specified in `--weights` (optional) and apply some post-processing.
    ```bash
    python -m technique_classification --config configs/tc_config.yml --create_submission_file
    ```
6. In case you have the correct markup in the `test_file` or gold `--test_labels_path` (source competition format), you can check your accuracy (micro f1-score) and f1-score per classes.
    ```bash
    python -m technique_classification --config configs/tc_config.yml  --eval_submission
    ```
7. Use `visualization_example/visualization.ipynb` if you want to visualize labels.

Our pretrained RoBERTa-CRF (SI task) and RoBERTa-Joined (TC task) models are available in [Google Drive](https://vk.com/away.php?to=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1Gph7FKMaxOBJdkrk0nM72uFpCGgn-2kC%3Fusp%3Dsharing).

## Citation

If you find this repository helpful, feel free to cite our publication [Aschern at SemEval-2020 Task 11: It Takes Three to Tango: RoBERTa, CRF, and Transfer Learning](https://www.aclweb.org/anthology/2020.semeval-1.191/):
```
@inproceedings{chernyavskiy-etal-2020-aschern,
    title = "Aschern at {S}em{E}val-2020 Task 11: It Takes Three to Tango: {R}o{BERT}a, {CRF}, and Transfer Learning",
    author = "Chernyavskiy, Anton  and
      Ilvovsky, Dmitry  and
      Nakov, Preslav",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.191",
    pages = "1462--1468"
}
``` 

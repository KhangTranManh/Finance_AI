---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: query
    dtype: string
  - name: label
    list: string
  - name: human_or_machine
    dtype:
      class_label:
        names:
          '0': human
          '1': machine
  splits:
  - name: train
    num_bytes: 1256446073.625
    num_examples: 28299
  - name: val
    num_bytes: 84202126
    num_examples: 1920
  - name: test
    num_bytes: 107049156.5
    num_examples: 2500
  download_size: 964095599
  dataset_size: 1447697356.125
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
license: gpl-3.0
---
# Dataset Card for "ChartQA"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
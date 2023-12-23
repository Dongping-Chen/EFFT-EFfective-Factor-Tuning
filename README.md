# EFFT-Effective-Factor-Tuning

This repository is a reproduction of the research presented in the paper ["Effective Factor Tuning"](https://arxiv.org/pdf/2311.06749). The goal is to replicate the study's findings and provide a transparent, open-source implementation for the community to explore and build upon.

## Description

Recent advancements have illuminated theefficacy of some tensorization-decomposition Parameter-Efficient Fine-Tuning methods like LoRA and F acT in the context of Vision Transformers (ViT). However, these methods grapple with the challenges of inadequately addressing inner- and cross-layer redundancy. To tackle this issue, we introduce **EF**fective **F**actor-**T**uning (**EFFT**), a simple yet effective fine-tuning method. Within the VTAB-1K dataset, our **EFFT** surpasses all baselines, attaining state-of-the-art performance with a categorical average of **75.9%** in top-1 accuracy with only **0.28%** of the parameters for full fine-tuning. Considering the simplicity and efficacy of **EFFT**, it holds the potential to serve as a foundational benchmark.

## Prerequisites

- Python 3.9
- Other dependencies specified in `requirements.txt`

## Installation

To set up your environment to run the code, follow these steps:

1. **Clone the Repository:**
<pre>
```shell
git clone https://github.com/Dongping-Chen/EFFT-EFfective-Factor-Tuning.git
cd EFFT-EFfective-Factor-Tuning
```
</pre>


2. **Create and Activate a Virtual Environment (optional but recommended) and Install the Required Packages:**
<pre>
```shell
conda activate EFFT
pip install -r requirements.txt
```
</pre>

## Usage

To execute the experiments, run:

<pre>
```shell
python execute.py
```
</pre>


### Parameters

You can customize the execution by specifying various parameters:

- `--model`: Choose between 'ViT' or 'Swin'.
- `--size`: For 'ViT', options include 'B', 'L', 'H'. For 'Swin', options include 'T', 'S', 'B', 'L'.
- `--dataset`: Select from a wide range of datasets including 'cifar', 'caltech101', 'dtd', and many others listed in the introduction.

Example:

<pre>
```shell
python execute.py --model "ViT" --size "B" --dataset "cifar"
```
</pre>


Note: When using the 'ViT B' model, optimal hyperparameters for replication will be automatically imported.

## Contributing

Contributions to this project are welcome. Please consider the following ways to contribute:

- Reporting issues
- Improving documentation
- Proposing new features or improvements

## License


## Acknowledgements

This project is based on the findings and methodologies presented in the paper ["Effective Factor Tuning"](https://arxiv.org/pdf/2311.06749). We would like to express our sincere appreciation to Tong Yao from Peking University (PKU) and Professor Yao Wan from Huazhong University of Science and Technology (HUST) for their invaluable contributions and guidance in this research.

## Citation

<pre>
```
@article{chen2023aggregate,
  title={Aggregate, Decompose, and Fine-Tune: A Simple Yet Effective Factor-Tuning Method for Vision Transformer},
  author={Chen, Dongping},
  journal={arXiv preprint arXiv:2311.06749},
  year={2023}
}
```
</pre>
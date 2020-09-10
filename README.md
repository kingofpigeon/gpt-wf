# GPT-WF (built on NVIDIA Megatron-LM)
This is the implementation for Chinese Web fiction Genrative Pre-trained model:GPT-WF.
## Quick start
### Installation
This repo is tested on Python 3.6+, PyTorch 1.5.0+.
To use the program the following prerequisites need to be installed.
- pytorch>=1.5.0
- nltk>=3.4
- numpy>=1.15.4
- pandas>=0.24.0
- sentencepiece>=0.1.8
- tensorflow>=1.12.0
- apex
### GPT-WF
##### Please download the pre-trained models from the following links and put it in folder ```~/Megatron-LM/checkpoints/generic/```
- Basic model: https://share.weiyun.com/r6Kdrm7T password：xqsuuv
- Stylized models https://share.weiyun.com/ELBwoIMU password：7gyavd

When you have downloaded the pre-trained models, you should run the following scripts to generate web fiction.
```
cd Megatron-LM
bash scripts/generate_text.sh
```
### Dataset 
For business reason, we don't present CWFC corpus. But we will present the corpus for fine-tuning stylized models soon.

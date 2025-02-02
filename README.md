# CIA
Code for training, evaluating and using a cross-lingual Auto Evaluator

## Installation

We require separate environments for training and evaluation due to incompatible Torch versions.

### Training Environment Setup

1. **Create and activate the training environment:**
   ```bash
   conda create -n training python=3.10 && conda activate training
   ```

2. **Install numpy (ensure compatibility by avoiding numpy 2.x):**
   ```bash
   pip install numpy==1.26.4
   ```

3. **Install PyTorch:**
   ```bash
   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

   - Use **torch v2.1.2** only.
   - Compile with CUDA based on your system specifications.
   - For further instructions, refer to the official [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

4. **Clone and install the Alignment Handbook:**
   ```bash
   git clone https://github.com/huggingface/alignment-handbook.git
   cd ./alignment-handbook/
   python -m pip install .
   ```

5. **Install Flash Attention 2:**
   ```bash
   python -m pip install flash-attn --no-build-isolation
   ```

6. **Login to Hugging Face CLI:**
   ```bash
   huggingface-cli login
   ```

7. **Install other useful libraries:**
   ```bash
   pip install wandb huggingface-hub==0.24.7
   ```

8. **Install Git LFS to push models to the Hugging Face Hub:**
   ```bash
   sudo apt-get install git-lfs
   ```


### Inference Environment Setup

1. **Create and activate the inference environment:**
   ```bash
   conda create -n inference python=3.10 && conda activate inference
   ```

2. **Install vLLM:**
   ```bash
   pip install vllm
   ```

3. **Install `datasets` and `transformers` libraries:**
   ```bash
   pip install datasets transformers
   ```

# Citation

If you find the following model helpful, please consider citing our paper!

**BibTeX:**

```bibtex
@article{doddapaneni2024crosslingual,
  title   = {Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs},
  author  = {Sumanth Doddapaneni and Mohammed Safi Ur Rahman Khan and Dilip Venkatesh and Raj Dabre and Anoop Kunchukuttan and Mitesh M. Khapra},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2410.13394}
}
```

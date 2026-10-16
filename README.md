# TAR: Targeted Adaptive Replay

Code for *Preserving Reasoning Diversity in RLVR via Targeted Adaptive Replay*.

## Training

### Environment

The experiments were run with Python 3.10, CUDA 12.4, torch 2.6.0, vllm 0.8.5.post1, transformers 4.51.1 and ray 2.44.1. Either of the below methods can set up the environment.

**Minimal install**

```bash
conda create -n tar python=3.10 -y && conda activate tar
pip install torch==2.6.0 vllm==0.8.5.post1 transformers==4.51.1
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install "ray[default]==2.44.1"
```

**Exact environment**

```bash
conda create -n tar python=3.10 -y && conda activate tar
pip install --no-deps -r recipe/tar/requirements-tar.txt
pip install --no-deps -e .
```

### Quick start with Qwen2.5-0.5B-Instruct

All commands below run in the `tar` environment.

```bash
# 1. data: MATH train (7.5k prompts) and MATH-500 (validation)
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
python3 recipe/tar/prepare_math500.py --local_dir ~/data/math

# 2. memory bank of the base policy
python3 recipe/tar/build_memory_bank.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output ~/data/math/tar_memory_bank_qwen2.5_0.5B.jsonl

# 3. train (1 node, 2 GPUs)
bash recipe/tar/run_qwen2_5_0.5b_tar.sh
```

## Evaluation

### Environment

```bash
conda create -n tar-eval python=3.9 -y && conda activate tar-eval
pip install --no-deps -r eval/requirements-eval.txt
pip install --no-deps -e eval/latex2sympy
```

### Evaluation
Convert the last checkpoint to Hugging Face format, then evaluate it (2 GPUs, tensor parallel):

```bash
CKPT=checkpoints/tar/qwen2.5_0.5B_tar/global_step_145/actor
python3 -m verl.model_merger merge --backend fsdp --local_dir $CKPT --target_dir $CKPT/hf   # tar environment

bash eval/run_eval.sh $CKPT/hf tar_0.5b   # tar-eval environment
```

## Acknowledgement

- The training code is built on [verl v0.4.1](https://github.com/volcengine/verl/tree/v0.4.1). (Apache 2.0).
- The accuracy evaluation is adapted from [R1_zero_Div](https://github.com/nigelyaoj/R1_zero_Div). 
- The n-gram tokenization of the diversity metrics follows [diversity-eval](https://github.com/GuyTevet/diversity-eval).

"""
Sample n responses per problem with vLLM and grade them.
Adapted from https://github.com/nigelyaoj/R1_zero_Div (MIT, see LICENSE).
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import TimeoutError

import numpy as np
import torch
from pebble import ProcessPool
from tqdm import tqdm
from transformers import AutoTokenizer

from grader import math_equal_process
from parser import STRIP_EXCEPTIONS, choice_answer_clean, extract_answer, parse_ground_truth, parse_question, strip_string

DATASETS = "gsm8k,math500,amc23,college_math,minerva_math,olympiadbench"
PROMPT = "{question} Let's think step by step and output the final answer within \\boxed{{}}."
STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]
EXTRA_KEYS = ["level", "type", "unit", "solution_type", "choices", "solution", "ques_type", "ans_type",
              "answer_type", "dataset", "subfield", "filed", "theorem", "answer"]
CHOICES = ["A", "B", "C", "D", "E"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_names", default=DATASETS)
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_data(data_name, data_dir):
    with open(f"{data_dir}/{data_name}/test.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    return sorted(examples, key=lambda x: x["idx"])


def is_multi_choice(answer):
    return all(c in CHOICES for c in answer)


def extract_pred(code, data_name, gt):
    if not code or code == "error":
        pred = "None"
    else:
        pred = strip_string(extract_answer(code, data_name), skip_unit=data_name in STRIP_EXCEPTIONS)
    if gt in CHOICES and pred not in CHOICES:
        pred = choice_answer_clean(code)
    elif is_multi_choice(gt) and not is_multi_choice(pred):
        pred = "".join(c for c in pred if c in CHOICES)
    return pred


def evaluate(samples, data_name):
    samples = sorted({s["idx"]: s for s in samples}.values(), key=lambda x: x["idx"])
    for sample in samples:
        sample["gt_cot"], sample["gt"] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample["gt"]) for idx, sample in enumerate(samples) for pred in sample["pred"]]

    scores, timeout_cnt = [], 0
    with ProcessPool(max_workers=1) as pool:
        iterator = pool.map(math_equal_process, params, timeout=3).result()
        for _ in tqdm(range(len(params)), desc="Evaluate"):
            try:
                scores.append(next(iterator))
            except TimeoutError:
                scores.append(False)
                timeout_cnt += 1

    i = 0
    for sample in samples:
        sample["score"] = scores[i : i + len(sample["pred"])]
        i += len(sample["pred"])
    score_array = np.array([s["score"] for s in samples])
    metrics = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s["pred"][-1]]),
        "acc": np.round(score_array.any(axis=1).mean() * 100, decimals=1),
        "sample_acc": list(np.round(score_array.mean(axis=0) * 100, decimals=1)),
    }
    return samples, metrics


def run(llm, tokenizer, sampling_params, data_name, args):
    samples = []
    for example in load_data(data_name, args.data_dir):
        question = parse_question(example, data_name)
        if question == "":
            continue
        gt_cot, gt = parse_ground_truth(example, data_name)
        sample = {"idx": example["idx"], "question": question, "gt_cot": gt_cot, "gt": gt}
        sample.update({k: example[k] for k in EXTRA_KEYS if k in example})
        samples.append(sample)

    prompts = [PROMPT.format(question=s["question"]).strip(" ") for s in samples for _ in range(args.n_sampling)]
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p.strip()}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    start_time = time.time()
    outputs = sorted(llm.generate(prompts, sampling_params), key=lambda x: int(x.request_id))
    codes = []
    for output in outputs:
        code = output.outputs[0].text.strip()
        for stop_word in STOP_WORDS:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
    time_use = time.time() - start_time

    n = args.n_sampling
    for i, sample in enumerate(samples):
        sample["code"] = codes[i * n : (i + 1) * n]
        sample["pred"] = [extract_pred(code, data_name, sample["gt"]) for code in sample["code"]]

    samples, metrics = evaluate(samples, data_name)
    metrics["time_use_in_second"] = time_use
    out_dir = f"{args.output_dir}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/samples.jsonl", "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4, default=float)
    print(data_name, metrics)
    return metrics


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_names = [
        d for d in args.data_names.split(",")
        if args.overwrite or not os.path.exists(f"{args.output_dir}/{d}/metrics.json")
    ]
    if not data_names:
        print("All datasets already evaluated.")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, dtype="auto", seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"] if t in tokenizer.get_vocab()]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        n=1,
        stop=STOP_WORDS,
        stop_token_ids=stop_token_ids or None,
    )
    for data_name in data_names:
        run(llm, tokenizer, sampling_params, data_name, args)


if __name__ == "__main__":
    main()

"""
Preprocess MATH-500 to parquet
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"
    dataset = datasets.load_dataset(data_source, split="test")
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": example["problem"] + " " + instruction_following}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {"split": "test", "index": idx, "level": example["level"], "subject": example["subject"]},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset.to_parquet(os.path.join(os.path.expanduser(args.local_dir), "test500.parquet"))

"""
Build the TAR memory bank
"""

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score


def farthest_point_sampling(embeddings: torch.Tensor, k: int) -> list:
    """Greedy FPS on L2-normalized embeddings, starting from the first sample."""
    n = embeddings.size(0)
    if n <= k:
        return list(range(n))
    selected = [0]
    min_dists = 1 - embeddings @ embeddings[0]
    min_dists[0] = -1.0
    while len(selected) < k:
        idx = int(torch.argmax(min_dists))
        selected.append(idx)
        min_dists = torch.minimum(min_dists, 1 - embeddings @ embeddings[idx])
        min_dists[selected] = -1.0
    return selected


@torch.no_grad()
def embed(texts, tokenizer, model, batch_size, max_length):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        out.append(F.normalize(model(**enc).last_hidden_state[:, 0], p=2, dim=1).float().cpu())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", default="~/data/math/train.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=128, help="samples per prompt")
    parser.add_argument("--k", type=int, default=3, help="targets kept per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--prompts_per_chunk", type=int, default=64)
    parser.add_argument("--embed_model", default="BAAI/bge-m3")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--embed_max_length", type=int, default=1024)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=None, help="only the first prompts (for testing)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(os.path.expanduser(args.train_file))
    if args.max_prompts is not None:
        df = df.iloc[: args.max_prompts]
    df = df.iloc[args.shard_id :: args.num_shards]

    progress_path = args.output + ".progress"
    done = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done = {int(line) for line in f if line.strip()}
    df = df[[int(e["index"]) not in done for e in df["extra_info"]]]
    print(f"[memory bank] {len(df)} prompts to process ({len(done)} done)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=-1, max_tokens=args.max_response_length)
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model = AutoModel.from_pretrained(args.embed_model).to("cuda").eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out_f, open(progress_path, "a") as progress_f:
        for start in range(0, len(df), args.prompts_per_chunk):
            chunk = df.iloc[start : start + args.prompts_per_chunk]
            prompts = []
            for messages in chunk["prompt"]:
                text = tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)
                ids = tokenizer(text, add_special_tokens=False).input_ids
                assert len(ids) <= args.max_prompt_length, f"prompt longer than {args.max_prompt_length} tokens"
                prompts.append({"prompt_token_ids": ids})
            generations = llm.generate(prompts, sampling_params)

            for (_, row), gen in zip(chunk.iterrows(), generations):
                gt = row["reward_model"]["ground_truth"]
                correct = [o.text for o in gen.outputs if float(default_compute_score(row["data_source"], o.text, gt)) == 1.0]
                if correct:
                    picked = farthest_point_sampling(embed(correct, embed_tokenizer, embed_model, args.embed_batch_size, args.embed_max_length), args.k)
                    for i in picked:
                        out_f.write(json.dumps({"index": int(row["extra_info"]["index"]), "output": correct[i]}, ensure_ascii=False) + "\n")
                progress_f.write(f"{int(row['extra_info']['index'])}\n")
            out_f.flush()
            progress_f.flush()
            print(f"[memory bank] {min(start + args.prompts_per_chunk, len(df))}/{len(df)} prompts")


if __name__ == "__main__":
    main()

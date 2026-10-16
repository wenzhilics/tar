"""
Diversity of the n responses of each problem, averaged over problems:
  div_equ: unique / total LaTeX equations (Div-Equ, Wu et al., 2025)
  ead:     expectation-adjusted distinct n-grams, averaged over n = 1..5 (EAD, Liu et al., 2022)
  emb_div: 1 - mean pairwise cosine similarity of BAAI/bge-m3 embeddings (Emb-Div)
The n-gram tokenization follows https://github.com/GuyTevet/diversity-eval (MIT).
"""

import argparse
import json
import os
import re
import warnings

import numpy as np

EQUATION_PATTERNS = (
    r"\$\$([\s\S]*?)\$\$",
    r"\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}",
    r"\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}",
    r"\\\[([\s\S]*?)\\\]",
    r"\\\(([\s\S]*?)\\\)",
    r"(?<!\$)\$(?!\$)((?:\\.|[^$\\])+?)\$(?!\$)",
)


def extract_equations(text):
    spans = []
    for pattern in EQUATION_PATTERNS:
        for m in re.finditer(pattern, text):
            inner = m.group(1).strip()
            if inner:
                spans.append((m.start(), m.end(), inner))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last_end = [], -1
    for start, end, inner in spans:
        if start >= last_end:
            out.append(inner)
            last_end = end
    return out


def div_equ(responses):
    equations = [eq for r in responses for eq in extract_equations(r)]
    return len(set(equations)) / len(equations) if equations else 1.0


def ead(responses, n_max=5, vocab_size=50257):
    words = [[w for w in r.replace(".", "").replace("\n", "").split(" ") if w != ""] for r in responses]
    values = []
    for n in range(1, n_max + 1):
        ngrams = [tuple(ws[i : i + n]) for ws in words for i in range(len(ws) - n + 1)]
        denom = vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** len(ngrams))
        values.append(len(set(ngrams)) / denom if denom else 0.0)
    return np.mean(values)


def emb_div(responses, model):
    from sklearn.metrics.pairwise import cosine_similarity

    sims = []
    if len(responses) >= 2:
        sim = cosine_similarity(model.encode(list(responses), batch_size=1024, show_progress_bar=False))
        sims = sim[np.triu_indices(len(responses), k=1)].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 1 - np.mean(sims)


def load_responses(path):
    response_sets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            responses = [c for c in json.loads(line)["code"] if c]
            if responses:
                response_sets.append(responses)
    return response_sets


def diversity(response_sets, model):
    per_problem = {"div_equ": [], "ead": [], "emb_div": []}
    for responses in response_sets:
        per_problem["div_equ"].append(div_equ(responses))
        per_problem["ead"].append(ead(responses))
        per_problem["emb_div"].append(emb_div(responses, model))
    result = {"num_problems": len(response_sets)}
    for k, v in per_problem.items():
        result[k] = float(np.nanmean(v))
        result[f"{k}_std"] = float(np.nanstd(v))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="output_dir of math_eval.py (pass@8 run)")
    parser.add_argument("--data_names", default="gsm8k,math500,amc23,college_math,minerva_math,olympiadbench")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    for data_name in args.data_names.split(","):
        out = f"{args.result_dir}/{data_name}/diversity.json"
        if os.path.exists(out) and not args.overwrite:
            continue
        result = diversity(load_responses(f"{args.result_dir}/{data_name}/samples.jsonl"), model)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(data_name, result)


if __name__ == "__main__":
    main()

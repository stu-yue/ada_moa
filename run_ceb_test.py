"""
Round-structured multi-agent debate with self-consistency + judge-driven
trajectory selection (an evolution of SMoA on the CEB benchmark).

Algorithm sketch
----------------
For every sample:

1. The debate is structured as ``ROUND_SETTING`` (hard-coded below). Each
   round declares
       * which reference models participate, and
       * how many candidate responses each of them must sample (``gen_num``).

2. In every round, for every reference model:
       a. The model produces ``gen_num`` candidates (self-consistency).
       b. The aggregator model judges those candidates and outputs both a
          ``best`` index and a ``second_best`` index.
       c. ``best`` is forwarded to the next round as cross-model context
          (the standard SMoA "references" channel).
       d. ``second_best`` is cached for the SAME model to consult in the
          NEXT round (sub-optimal trajectory injection).

3. After every round (except the last), the aggregator may decide to end
   the debate early via ``--moderate_end``.

4. The aggregator integrates the final-round best responses into one
   answer.

Naming
------
All model identifiers are FULL HuggingFace names (e.g. ``"Qwen/Qwen3-8B"``)
because the rest of the codebase (vLLM endpoint resolution, tokenizer
loading, ``token_num_dict`` keys) uses full names. Only the OUTPUT FILENAME
truncates to the tail with ``model.split('/')[1]``.
"""

import os
import json
import copy
import re
import random
import argparse
from functools import partial
import warnings

import datasets
from transformers import AutoTokenizer

from utils import (
    generate_vllm,
    generate_with_references,
    inject_references_to_messages,
    inject_role_to_messages,
    extract_role_from_output,
    get_tokenizer_name,
)

random.seed(42)


ROUND_SETTING = {
    0: {"ref_models": ["Qwen/Qwen3-4B", "HuggingFaceTB/SmolLM3-3B"], "gen_num": 5},
    1: {"ref_models": ["Qwen/Qwen3-4B, Qwen/Qwen3-8B", "HuggingFaceTB/SmolLM3-3B"], "gen_num": 3},
    2: {"ref_models": ["Qwen/Qwen3-8B"], "gen_num": 1},
}


JUDGE_PROMPT_TEMPLATE = (
    "You are an expert judge selecting among {K} candidate responses to the "
    "same question. Read every candidate carefully and identify the BEST and "
    "the SECOND-BEST one.\n\n"
    "Question:\n{question}\n\n"
    "{candidates}\n\n"
    "Output STRICTLY in the following format and nothing else:\n"
    '"best": <index>\n'
    '"second_best": <index>\n'
)


END_DEBATE_PROMPT_TEMPLATE = (
    "You are a moderator deciding whether the multi-agent debate should END.\n\n"
    "Question:\n{question}\n\n"
    "Best response from each agent in the current round:\n{responses}\n\n"
    "If these responses already converge and additional rounds would not "
    "meaningfully improve the answer, end the debate.\n"
    "Output STRICTLY in the following format and nothing else:\n"
    '"end debate": <True or False>\n'
)


SECOND_BEST_INJECT_TEMPLATE = (
    "Note: in your previous round, you yourself produced the following "
    "sub-optimal trajectory for the same question. You may take it as an "
    "auxiliary reference, but feel free to disagree with or deviate from it:\n\n"
    "<previous_sub_optimal_trajectory>\n{sub_optimal}\n</previous_sub_optimal_trajectory>"
)


# =============================================================================
# Helpers
# =============================================================================

def validate_round_setting(round_setting):
    """Cheap structural / type / naming check on ``ROUND_SETTING``."""
    assert isinstance(round_setting, dict) and round_setting, \
        "ROUND_SETTING must be a non-empty dict."
    keys = sorted(round_setting.keys())
    assert keys == list(range(len(round_setting))), (
        f"ROUND_SETTING keys must be contiguous 0..N-1, got {keys}."
    )
    for r, cfg in round_setting.items():
        assert {"ref_models", "gen_num"} <= cfg.keys(), (
            f"round {r}: each entry must contain 'ref_models' and 'gen_num'."
        )
        assert isinstance(cfg["ref_models"], list) and cfg["ref_models"], (
            f"round {r}: 'ref_models' must be a non-empty list."
        )
        assert isinstance(cfg["gen_num"], int) and cfg["gen_num"] >= 1, (
            f"round {r}: 'gen_num' must be a positive int."
        )
        for m in cfg["ref_models"]:
            assert isinstance(m, str) and "/" in m, (
                f"round {r}: model name '{m}' should be a full HF identifier "
                "of the form 'org/name' (do NOT split on '/')."
            )
        assert len(set(cfg["ref_models"])) == len(cfg["ref_models"]), (
            f"round {r}: duplicate models within a single round are disallowed."
        )


def collect_models_from_round_setting(round_setting):
    """Return the de-duplicated, order-preserving union of all ref models."""
    seen, out = set(), []
    for r in sorted(round_setting.keys()):
        for m in round_setting[r]["ref_models"]:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def extract_best_and_second_best(output, k):
    """Parse ``"best": i`` / ``"second_best": j`` from a judge output.

    Returns
    -------
    (best_idx, sec_best_idx) where each is either an int in ``[0, k)`` or None.
    """
    if output is None:
        return None, None
    best_match = re.search(r'"best"\s*:\s*(\d+)', output)
    sec_match = re.search(r'"second_best"\s*:\s*(\d+)', output)
    best = int(best_match.group(1)) if best_match else None
    sec = int(sec_match.group(1)) if sec_match else None
    if best is not None and not (0 <= best < k):
        best = None
    if sec is not None and not (0 <= sec < k):
        sec = None
    return best, sec


def extract_end_debate(output):
    """Parse ``"end debate": True/False`` from moderator output."""
    if output is None:
        return False
    m = re.search(r'"end\s*debate"\s*:\s*(True|False|true|false)', output)
    return bool(m and m.group(1).lower() == "true")


def inject_sub_optimal_trajectory(messages, sub_optimal):
    """Fold the same-model previous-round sub-optimal trajectory into the
    system message (or prepend a new system message)."""
    messages = copy.deepcopy(messages)
    block = SECOND_BEST_INJECT_TEMPLATE.format(sub_optimal=sub_optimal)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + block
    else:
        messages = [{"role": "system", "content": block}] + messages
    return messages


def build_candidate_block(candidates):
    return "\n\n".join(f"Response {i}.\n{c}" for i, c in enumerate(candidates))


# =============================================================================
# Token bookkeeping
# =============================================================================

def _init_token_dict(model, all_ref_models):
    """Aggregator key always present and bucketed; reference models are flat."""
    bucketed = {"debate": 0, "judge": 0, "moderate": 0, "aggregate": 0}
    out = {}
    for m in all_ref_models:
        out[m] = copy.deepcopy(bucketed) if m == model else 0
    if model not in out:
        out[model] = copy.deepcopy(bucketed)
    return out


def _count_msgs(tokenizer, msgs):
    return sum(len(tokenizer.tokenize(m["content"])) for m in msgs)


def _count_text(tokenizer, text):
    return len(tokenizer.tokenize(text)) if text else 0


# =============================================================================
# Core
# =============================================================================

def process_fn(
    item,
    *,
    model,
    args,
    round_setting,
    tokenizer_dict,
    temperature=0.7,
    max_tokens=2048,
):
    question = item["question"]
    all_ref_models = collect_models_from_round_setting(round_setting)
    token_num_dict = _init_token_dict(model, all_ref_models)

    # ---- token-accounting closures (capture token_num_dict / tokenizer_dict) ----

    def bill_ref(m, prompt_msgs, completions):
        tk = tokenizer_dict[m]
        n_in = _count_msgs(tk, prompt_msgs)
        n_out = sum(_count_text(tk, c) for c in completions)
        if m == model:
            token_num_dict[m]["debate"] += n_in + n_out
        else:
            token_num_dict[m] += n_in + n_out

    def bill_aggregator(bucket, prompt_msgs, output_text):
        tk = tokenizer_dict[model]
        token_num_dict[model][bucket] += _count_msgs(tk, prompt_msgs) + _count_text(tk, output_text)

    # ---- (optional) per-round role-description prep ----

    role_lists = {}
    if args.add_role:
        for r, cfg in round_setting.items():
            n_models = len(cfg["ref_models"])
            cache_path = f"prompt/{args.dataset}/role_description_n{n_models}.json"
            roles = []
            if os.path.exists(cache_path):
                roles = json.load(open(cache_path))
            if len(roles) < n_models:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                msgs = [{
                    "role": "user",
                    "content": args.role_generation_prompt.format(n_models, n_models, args.task),
                }]
                role_out, _ = generate_vllm(
                    messages=msgs, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                )
                if role_out is not None:
                    roles = extract_role_from_output(role_out)
                    with open(cache_path, "w") as f:
                        json.dump(roles, f, indent=2)
            role_lists[r] = roles[:n_models] if roles else [""] * n_models

    # ---- main debate loop ----

    internal_result = {m: {} for m in all_ref_models}
    internal_result["__moderation__"] = {}

    prev_best_responses = []   # cross-model "references" forwarded to next round
    final_round = -1
    total_rounds = len(round_setting)

    for r in range(total_rounds):
        cfg = round_setting[r]
        ref_models = cfg["ref_models"]
        gen_num = cfg["gen_num"]
        roles = role_lists.get(r, [""] * len(ref_models))

        round_best_responses = []

        for idx, ref_model in enumerate(ref_models):

            # Build prompt: question + cross-model refs + own sub-optimal + role
            msgs = [{"role": "user", "content": question}]
            if prev_best_responses:
                msgs = inject_references_to_messages(
                    msgs, prev_best_responses, args.aggreagator_system_prompt,
                )
            if r > 0:
                sub_optimal = (
                    internal_result.get(ref_model, {})
                    .get(f"round_{r - 1}", {})
                    .get("sec_best")
                )
                if sub_optimal:
                    msgs = inject_sub_optimal_trajectory(msgs, sub_optimal)
            if args.add_role and idx < len(roles) and roles[idx]:
                msgs = inject_role_to_messages(msgs, roles[idx])

            # Self-consistency: K candidates (single batched vLLM call when K>1)
            out, _ = generate_vllm(
                messages=msgs, model=ref_model,
                temperature=temperature, max_tokens=max_tokens, n=gen_num,
            )
            if out is None:
                candidates = []
            elif isinstance(out, str):
                candidates = [out]
            else:
                candidates = list(out)

            if not candidates:
                internal_result[ref_model][f"round_{r}"] = {
                    "candidates": [], "best": None, "sec_best": None,
                    "best_idx": None, "sec_best_idx": None, "judge_raw": None,
                }
                warnings.warn(f"No candidates generated for {ref_model} in round {r}")
                continue

            bill_ref(ref_model, msgs, candidates)

            # Aggregator-as-judge over the K candidates
            if len(candidates) == 1:
                best_idx, sec_best_idx, judge_raw = 0, None, None
            else:
                judge_msgs = [{
                    "role": "user",
                    "content": JUDGE_PROMPT_TEMPLATE.format(
                        K=len(candidates),
                        question=question,
                        candidates=build_candidate_block(candidates),
                    ),
                }]
                judge_raw, judge_input = generate_vllm(
                    messages=judge_msgs, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                )
                bill_aggregator("judge", judge_input or judge_msgs, judge_raw or "")

                best_idx, sec_best_idx = extract_best_and_second_best(judge_raw, len(candidates))
                if best_idx is None:
                    best_idx = random.randrange(len(candidates))
                if sec_best_idx is None or sec_best_idx == best_idx:
                    others = [i for i in range(len(candidates)) if i != best_idx]
                    sec_best_idx = random.choice(others) if others else None

            best_resp = candidates[best_idx]
            sec_best_resp = candidates[sec_best_idx] if sec_best_idx is not None else None

            internal_result[ref_model][f"round_{r}"] = {
                "candidates": candidates,
                "best": best_resp,
                "sec_best": sec_best_resp,
                "best_idx": best_idx,
                "sec_best_idx": sec_best_idx,
                "judge_raw": judge_raw,
            }
            round_best_responses.append(best_resp)

        # Moderate-end check (skip on the last round - nothing to early-stop)
        ended = False
        if (
            args.moderate_end
            and round_best_responses
            and r < total_rounds - 1
        ):
            try:
                end_msgs = [{
                    "role": "user",
                    "content": END_DEBATE_PROMPT_TEMPLATE.format(
                        question=question,
                        responses=build_candidate_block(round_best_responses),
                    ),
                }]
                end_raw, end_input = generate_vllm(
                    messages=end_msgs, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                )
                bill_aggregator("moderate", end_input or end_msgs, end_raw or "")
                ended = extract_end_debate(end_raw)
                internal_result["__moderation__"][f"round_{r}"] = {
                    "raw": end_raw, "ended": ended,
                }
            except Exception as e:
                print(e)
                internal_result["__moderation__"][f"round_{r}"] = {
                    "raw": None, "ended": False, "error": str(e),
                }

        prev_best_responses = round_best_responses
        final_round = r
        if ended:
            break

    # ---- final aggregation ----

    final_msgs = [{"role": "user", "content": question}]
    final_output, final_input = generate_with_references(
        model=model,
        messages=final_msgs,
        references=prev_best_responses,
        system=args.aggreagator_system_prompt,
        generate_fn=generate_vllm,
    )

    if final_output is not None:
        bill_aggregator("aggregate", final_input or final_msgs, final_output)

    return {
        "response": json.dumps({
            "choices": [{"message": {"content": final_output, "role": "assistant"}}]
        }),
        "internal_result": internal_result,
        "generator": model + "-vllm",
        "judge_output": None,
        "chosen_responses": None,
        "token_num_dict": token_num_dict,
        "total_round": final_round,
    }


# =============================================================================
# Entry
# =============================================================================

def generate_for_ceb(args):
    validate_round_setting(ROUND_SETTING)

    args.aggreagator_system_prompt = open(
        f"prompt/{args.dataset}/aggreagator_system_prompt.txt"
    ).read()
    if args.add_role:
        args.role_generation_prompt = open(
            f"prompt/{args.dataset}/role_generation_prompt_v2.txt"
        ).read()
        args.task = open(f"prompt/{args.dataset}/task.txt").read()

    all_ref_models = collect_models_from_round_setting(ROUND_SETTING)
    all_models = list(dict.fromkeys(all_ref_models + [args.model]))
    tokenizer_dict = {
        m: AutoTokenizer.from_pretrained(get_tokenizer_name(m)) for m in all_models
    }

    eval_set = []
    for file in os.listdir(f"data/{args.dataset}"):
        data = json.load(open(os.path.join(f"data/{args.dataset}", file)))
        for line in data:
            eval_set.append(line)

    sample_num = min(args.sample_num, len(eval_set))
    eval_set = random.sample(eval_set, sample_num)

    eval_set = {
        **{key: [item[key] for item in eval_set] for key in eval_set[0].keys()},
        **{"question": [item["prompt"] for item in eval_set]},
    }
    eval_set = datasets.Dataset.from_dict(eval_set)

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=args.model,
            round_setting=ROUND_SETTING,
            tokenizer_dict=tokenizer_dict,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            args=args,
        ),
        batched=False,
        num_proc=args.num_proc,
    )

    output_folder = f"output/{args.dataset}"
    os.makedirs(output_folder, exist_ok=True)

    model_short = args.model.split("/")[1]
    output_path = os.path.join(
        output_folder,
        "alg-roundjudge_model-{m}_rounds-{r}_nrefs-{n}_addrole-{ar}_modend-{me}.jsonl".format(
            m=model_short,
            r=len(ROUND_SETTING),
            n=len(all_ref_models),
            ar=args.add_role,
            me=args.moderate_end,
        ),
    )

    with open(output_path, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-structured SC+judge debate on CEB")

    parser.add_argument("--dataset", type=str, default="CEB-Conversation-S",
                        help="dataset to use")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-72B-Instruct",
                        help="aggregator / judge model (full HF name)")
    parser.add_argument("--num_proc", type=int, default=6,
                        help="datasets.map worker count")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--sample_num", type=int, default=200)

    parser.add_argument("--add_role", action="store_true",
                        help="auto-generate per-round role descriptions for ref models")
    parser.add_argument("--moderate_end", action="store_true",
                        help="let aggregator early-stop the debate after each round")

    args = parser.parse_args()
    generate_for_ceb(args)

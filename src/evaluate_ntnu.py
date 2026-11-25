"""Generate and blind-score the offline NTNU question set."""

import argparse
import json
from collections import Counter
from hashlib import file_digest, sha256
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from device_utils import add_device_argument, parse_4bit, resolve_device
from inference import load_base_model_and_tokenizer, resolve_adapter_path

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "evaluation" / "ntnu_qa.jsonl"
CATEGORY_TOTALS = {"fact": 40, "uncertain": 10, "general": 10}


def read_cases(path: Path) -> list[dict[str, Any]]:
    """Read the frozen questions and their grading criteria.

    Args:
        path (Path): JSONL file containing the test cases.

    Returns:
        list[dict[str, Any]]: Test cases ordered as in the file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the IDs or category totals are invalid.
    """
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        Counter(case["category"] for case in cases) != CATEGORY_TOTALS
        or len({case["id"] for case in cases}) != len(cases)
    ):
        raise ValueError("Expected 60 unique questions in the 40/10/10 categories.")
    return cases


def write_json(path: Path, content: Any) -> None:
    """Write a readable JSON result.

    Args:
        path (Path): Destination path.
        content (Any): JSON-serializable value.

    Returns:
        None.

    Raises:
        OSError: If the result cannot be written.
        TypeError: If the result is not JSON-serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False,
    ) as file:
        pending = Path(file.name)
        try:
            json.dump(content, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.close()
            pending.replace(path)
        finally:
            file.close()
            pending.unlink(missing_ok=True)


def generate_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate greedy answers using the base model's inference template.

    Args:
        model (torch.nn.Module): Base model with its trained adapter.
        tokenizer (PreTrainedTokenizerBase): Tokenizer with left padding enabled.
        prompts (list[str]): User prompts without grading criteria.
        max_new_tokens (int): Common completion limit.

    Returns:
        list[str]: Answers in the same order as the prompts.

    Raises:
        RuntimeError: If inference fails.
    """
    wrapped = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(wrapped, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [
        answer.strip()
        for answer in tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
    ]


def generate(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
    """Load one adapter and generate resumable answers.

    Args:
        args (argparse.Namespace): Model, adapter, generation and output options.
        cases (list[dict[str, Any]]): Questions to answer.

    Returns:
        None.

    Raises:
        ValueError: If an existing output belongs to a different run.
        RuntimeError: If the model cannot generate answers.
    """
    args.torch_dtype = "bfloat16"
    if args.bf16_weights:
        args.load_in_4bit = False
    adapter = resolve_adapter_path(args.adapter_path)
    weights = Path(adapter) / "adapter_model.safetensors"
    if not weights.is_file():
        weights = Path(adapter) / "adapter_model.bin"
    with weights.open("rb") as file:
        adapter_hash = file_digest(file, "sha256").hexdigest()
    legacy_run = {
        "model": args.model_id,
        "adapter": adapter,
        "adapter_weights_sha256": adapter_hash,
        "cases_sha256": sha256(args.cases.read_bytes()).hexdigest(),
        "bf16_weights": args.bf16_weights,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }
    config = resolve_device(args)
    run = {
        **legacy_run,
        "backend": config.backend,
        "dtype": config.dtype_name,
        "load_in_4bit": config.load_in_4bit,
    }
    answers = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        # The old format only recorded the intended CUDA/BF16 preset.
        matches_legacy = (
            previous["run"] == legacy_run
            and config.backend == "cuda"
            and config.dtype == torch.bfloat16
            and config.load_in_4bit == (not args.bf16_weights)
        )
        if previous["run"] != run and not matches_legacy:
            raise ValueError("The output belongs to a different evaluation run.")
        answers = previous["answers"]
    remaining = [case for case in cases if case["id"] not in answers]
    if not remaining:
        return
    model, tokenizer = load_base_model_and_tokenizer(
        args.model_id, config.load_in_4bit, config.dtype, config.device_map,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer.padding_side = "left"
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start:start + args.batch_size]
        completions = generate_batch(
            model, tokenizer, [case["prompt"] for case in batch], args.max_new_tokens,
        )
        answers.update(zip((case["id"] for case in batch), completions, strict=True))
        write_json(args.output, {"run": run, "answers": answers})
        print(f"Generated {len(answers)}/{len(cases)} answers", flush=True)


def blind(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
    """Shuffle each question's outputs independently and save a separate key.

    Args:
        args (argparse.Namespace): Named responses, blind output and key paths.
        cases (list[dict[str, Any]]): Questions and grading criteria.

    Returns:
        None.

    Raises:
        ValueError: If run names are duplicated or a run has missing answers.
    """
    runs = {}
    for entry in args.responses:
        name, separator, path = entry.partition("=")
        if not separator or not name or name in runs:
            raise ValueError("Use a unique NAME=PATH for every response file.")
        answers = json.loads(Path(path).read_text(encoding="utf-8"))["answers"]
        if answers.keys() != {case["id"] for case in cases}:
            raise ValueError(f"Run {name} does not have exactly one answer per case.")
        runs[name] = answers
    random = Random(42)
    review, key = [], {}
    for case in cases:
        names = sorted(runs)
        random.shuffle(names)
        key[case["id"]] = names
        review.append({
            **case,
            "answers": [
                {
                    "text": runs[name][case["id"]],
                    "pass": None, "major": None, "note": "",
                }
                for name in names
            ],
        })
    write_json(args.output, review)
    write_json(args.key, key)


def score(args: argparse.Namespace, cases: list[dict[str, Any]]) -> None:
    """Summarize the blind review and apply the quality gates.

    Args:
        args (argparse.Namespace): Scored review, key and two baseline names.
        cases (list[dict[str, Any]]): Original questions and category assignments.

    Returns:
        None.

    Raises:
        ValueError: If a grade is missing or the review and key disagree.
    """
    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.key.read_text(encoding="utf-8"))
    categories = {case["id"]: case["category"] for case in cases}
    if len(review) != len(cases) or {row["id"] for row in review} != categories.keys():
        raise ValueError("The review does not cover exactly the fixed test questions.")
    results = {}
    for row in review:
        case_id = row["id"]
        for name, answer in zip(key[case_id], row["answers"], strict=True):
            passed, major = answer["pass"], answer["major"]
            if not isinstance(passed, bool) or not isinstance(major, bool):
                raise ValueError(f"Reviewer must grade pass and major for {case_id}.")
            if passed and major:
                raise ValueError(f"{case_id} cannot pass with a major fabrication.")
            totals = results.setdefault(
                name,
                {"fact": 0, "uncertain": 0, "general": 0, "major": 0, "misses": []},
            )
            totals[categories[case_id]] += passed
            totals["major"] += major
            if not passed:
                totals["misses"].append({"id": case_id, "note": answer["note"]})
    baseline_names = set(args.baselines)
    if len(baseline_names) != 2 or not baseline_names <= results.keys():
        raise ValueError("Both named comparison baselines must appear in the review.")
    min_fact = max(34, *(results[name]["fact"] for name in args.baselines))
    min_general = max(results[name]["general"] for name in args.baselines) - 1
    for name, totals in results.items():
        if name not in args.baselines:
            totals["passes_quality"] = (
                totals["fact"] >= min_fact
                and totals["uncertain"] >= 8
                and totals["general"] >= min_general
                and totals["major"] == 0
            )
    write_json(args.output, results)


def main() -> None:
    """Dispatch response generation, review preparation or quality scoring.

    Returns:
        None.

    Raises:
        ValueError: If inputs or grades are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    commands = parser.add_subparsers(required=True)
    generation = commands.add_parser("generate")
    generation.set_defaults(action=generate)
    generation.add_argument("--model_id", required=True)
    generation.add_argument("--adapter_path", required=True)
    add_device_argument(generation, default="gpu")
    weights = generation.add_mutually_exclusive_group()
    weights.add_argument("--bf16_weights", action="store_true",
                         help="Use unquantized weights with BF16 when supported")
    weights.add_argument("--load_in_4bit", type=parse_4bit,
                         help="Use NF4 (enabled by default on NVIDIA CUDA)")
    generation.add_argument("--batch_size", type=int, default=4)
    generation.add_argument("--max_new_tokens", type=int, default=192)
    generation.add_argument("--output", type=Path, required=True)
    blinding = commands.add_parser("blind")
    blinding.set_defaults(action=blind)
    blinding.add_argument("responses", nargs="+", metavar="NAME=PATH")
    blinding.add_argument("--output", type=Path, required=True)
    blinding.add_argument("--key", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.set_defaults(action=score)
    scoring.add_argument("--review", type=Path, required=True)
    scoring.add_argument("--key", type=Path, required=True)
    scoring.add_argument("--baselines", nargs=2, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action is generate and (args.batch_size < 1 or args.max_new_tokens < 1):
        parser.error("Batch size and completion cap must be positive.")
    args.action(args, read_cases(args.cases))


if __name__ == "__main__":
    main()

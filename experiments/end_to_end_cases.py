from __future__ import annotations

import time
import os
import csv
from pathlib import Path

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder


try:
    from bencheval.tabarena import TabArena
except ModuleNotFoundError:
    from experiments.bencheval.tabarena import TabArena
from experiments.dataset_utils import (
    pre_process_and_split_data,
    pre_process_and_split_data_dp_range_zscore_with_public_ranges,
    pre_process_and_split_data_public_minmax,
)

import json
import re
import random
from typing import Any

import numpy as np
import torch
import joblib

# Choose one with LLM_PROVIDER=qwen or LLM_PROVIDER=gemini.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
# QWEN_MODEL_ID = "Qwen/Qwen3-1.7B"
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "1"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
BANK_MARKETING_OPENML_ID = int(os.getenv("BANK_MARKETING_OPENML_ID", "44234"))
PIMA_DIABETES_OPENML_ID = int(os.getenv("PIMA_DIABETES_OPENML_ID", "37"))
STUDENT_PERFORMANCE_OPENML_ID = int(os.getenv("STUDENT_PERFORMANCE_OPENML_ID", "46589"))
COMPAS_OPENML_ID = int(os.getenv("COMPAS_OPENML_ID", "42192"))
ACS_INCOME_MAX_ROWS = int(os.getenv("ACS_INCOME_MAX_ROWS", "50000"))
ACS_INCOME_SEED = int(os.getenv("ACS_INCOME_SEED", "0"))
MATERNAL_HEALTH_RISK_UCI_ID = int(os.getenv("MATERNAL_HEALTH_RISK_UCI_ID", "863"))
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", "/tmp/privtab_hf_datasets_cache")
BOUNDS_CACHE_DIR = Path(os.getenv("BOUNDS_CACHE_DIR", "experiments/bounds_cache"))
PROMPTS_CACHE_DIR = Path(os.getenv("PROMPTS_CACHE_DIR", "experiments/prompts_cache"))
REFRESH_LLM_BOUNDS = os.getenv("REFRESH_LLM_BOUNDS", "0") == "1"
CASE_STUDY_REPORT_PATH = Path(
    os.getenv("CASE_STUDY_REPORT_PATH", "experiments/end_to_end_case_report.md")
)
CASE_STUDY_RESULTS_CSV_PATH = Path(
    os.getenv("CASE_STUDY_RESULTS_CSV_PATH", "experiments/end_to_end_case_results.csv")
)
END_TO_END_DP_BASELINES_RESULTS = os.getenv("END_TO_END_DP_BASELINES_RESULTS", "")
END_TO_END_INDICES_DIR = os.getenv(
    "END_TO_END_INDICES_DIR",
    os.getenv(
        "END_TO_END_DP_BASELINES_INDICES_DIR",
        os.path.join(os.environ.get("SCRATCH_PATH", "experiments"), "end_to_end_indices"),
    ),
)
END_TO_END_ASINH_NUMERIC = os.getenv("END_TO_END_ASINH_NUMERIC", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_MODEL_SWITCH_THRESHOLD = 4096
DEFAULT_SMALL_CONTEXT_WEIGHTS = "models/stage2_short_contexts/weights.pt"
DEFAULT_LARGE_CONTEXT_WEIGHTS = "models/stage2_long_contexts/weights.pt"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_end_to_end_indices_with_class_coverage(y, train_fraction, seed=0):
    y = np.asarray(y)
    n = len(y)
    indices = np.arange(n)
    train_size = int(n * train_fraction)
    classes = np.unique(y)

    if train_size < len(classes):
        raise ValueError(
            f"Train size {train_size} is smaller than number of classes {len(classes)}."
        )

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for cls in classes:
        cls_idx = indices[y == cls].copy()
        rng.shuffle(cls_idx)
        cls_train_size = int(round(len(cls_idx) * train_fraction))
        cls_train_size = max(1, min(cls_train_size, len(cls_idx) - 1))
        train_parts.append(cls_idx[:cls_train_size])
        test_parts.append(cls_idx[cls_train_size:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    while len(train_idx) > train_size:
        moved = train_idx[-1:]
        train_idx = train_idx[:-1]
        test_idx = np.concatenate([test_idx, moved])
    while len(train_idx) < train_size and len(test_idx) > 0:
        moved = test_idx[-1:]
        test_idx = test_idx[:-1]
        train_idx = np.concatenate([train_idx, moved])

    return np.concatenate([np.sort(train_idx), np.sort(test_idx)])


def ensure_end_to_end_indices(
        y,
        dataset_slug: str,
        indices_dir: str,
        repeats: int,
        train_fraction: float,
):
    os.makedirs(indices_dir, exist_ok=True)
    for repeat in range(repeats):
        path = os.path.join(indices_dir, f"{dataset_slug}_repeat_{repeat}.pickle")
        if not os.path.isfile(path):
            indices = split_end_to_end_indices_with_class_coverage(
                y,
                train_fraction=train_fraction,
                seed=repeat,
            )
            joblib.dump(indices, filename=path)


def load_end_to_end_indices(indices_dir: str, dataset_slug: str, repeat: int):
    path = os.path.join(indices_dir, f"{dataset_slug}_repeat_{repeat}.pickle")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split indices file: {path}")
    return joblib.load(path)


def load_qwen_model(model_id: str = QWEN_MODEL_ID, local_files_only: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError(
            "Qwen support requires the transformers package. "
            "Install it or use cached bounds / LLM_PROVIDER=gemini."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only,
    )

    model.eval()
    return tokenizer, model


def load_gemini_client():
    try:
        from google import genai
    except ImportError as error:
        raise ImportError(
            "Gemini support requires the google-genai package. "
            "Install it with `pip install google-genai`."
        ) from error

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it in the environment before "
            "using LLM_PROVIDER=gemini."
        )

    # genai.Client reads GEMINI_API_KEY from the environment.
    return genai.Client()


def chunks(items: list[dict[str, str]], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i: i + chunk_size]


def safe_cache_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_batch_messages(
        columns: list[dict[str, str]],
        dataset_context: str,
) -> list[dict[str, str]]:
    system_prompt = """
You are extracting PUBLIC clipping bounds for differentially private preprocessing.

Return strict JSON only.
Do not include markdown.
Do not include explanations outside JSON.
"""

    columns_text = "\n".join(
        "\n".join(
            line
            for line in (
                f"{i + 1}. name: {col['name']}",
                f"   description: {col['description']}",
                f"   unit: {col['unit']}" if col.get("unit") else None,
            )
            if line is not None
        )
        for i, col in enumerate(columns)
    )

    user_prompt = f"""
Rules:
1. Analyze each target column independently.
2. Do NOT claim exact bounds unless they are explicit or follow from definitions.
3. Do NOT use empirical min/max values unless explicitly provided in the public description.
4. If true documented bounds are unavailable, you MUST still provide finite suggested clipping bounds.
5. The suggested clipping bounds must be conservative and based on public semantics, units, and dataset context.
6. Clearly distinguish documented bounds from guessed clipping bounds.
7. Use null for documented bounds that are not justified.
8. Always explain the evidence and the guess rationale.
9. Return one JSON object per target column.
10. Do not include columns that were not provided.
11. If a public semantic lower bound is clear, include it even when the upper bound is not documented.
12. All suggested clipping bounds must be finite numeric values.
13. Do not omit a target column. If uncertain, use documented_bound_status="insufficient_information" and suggested_clip_status="chosen_clipping_bound".
14. The top-level JSON value must be an array, not an object wrapper.
15. Use the dataset-specific guidance in the Dataset context when it defines units,
    special values, or semantic lower/upper bounds.

Bound evidence statuses:
- exact_public_bound: exact range follows from definition or description, e.g. day of month = 1..31.
- documented_special_value: special value is documented.
- semantic_public_bound: public semantics strongly imply the bound.
- insufficient_information: description does not justify a documented numeric bound.

Return strict JSON only as an array:

[
  {{
    "feature": string,
    "unit": string | null,

    "documented_lower_bound": number | null,
    "documented_upper_bound": number | null,
    "documented_bound_status": "exact_public_bound" | "documented_special_value" | "semantic_public_bound" | "insufficient_information",

    "suggested_lower_clip": number,
    "suggested_upper_clip": number,
    "suggested_clip_status": "exact_public_bound" | "semantic_public_bound" | "chosen_clipping_bound",

    "evidence": string,
    "guess_rationale": string,
    "confidence": "low" | "medium" | "high"
  }}
]

Dataset context:
{dataset_context}

Target columns:
{columns_text}
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def apply_chat_template(
        tokenizer,
        messages: list[dict[str, str]],
        model_id: str,
) -> str:
    """
    Qwen3 supports enable_thinking=False.
    Qwen2.5 does not need it.
    """
    if "Qwen3" in model_id:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_json_results(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        if "feature" in parsed:
            return [parsed]

        for key in ("results", "bounds", "columns", "features", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value

        list_values = [
            value for value in parsed.values()
            if isinstance(value, list)
        ]
        if len(list_values) == 1:
            return list_values[0]

        return [parsed]

    raise ValueError(f"Expected a JSON array or object, got {type(parsed).__name__}")


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()

    # Remove common markdown fences if the model adds them.
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return normalize_json_results(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON array.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output:\n{text}")

    return normalize_json_results(json.loads(match.group(0)))


def extract_gemini_json_array(response: Any) -> list[dict[str, Any]]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return normalize_json_results(parsed)

    text = getattr(response, "text", None)
    if not text:
        raise ValueError(f"Gemini returned no text or parsed JSON: {response!r}")

    try:
        return extract_json_array(text)
    except Exception as error:
        raise ValueError(f"Failed to parse Gemini JSON response:\n{text}") from error


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "feature",
        "unit",
        "documented_lower_bound",
        "documented_upper_bound",
        "documented_bound_status",
        "suggested_lower_clip",
        "suggested_upper_clip",
        "suggested_clip_status",
        "evidence",
        "guess_rationale",
        "confidence",
    }

    allowed_documented_statuses = {
        "exact_public_bound",
        "documented_special_value",
        "semantic_public_bound",
        "insufficient_information",
    }

    allowed_clip_statuses = {
        "exact_public_bound",
        "semantic_public_bound",
        "chosen_clipping_bound",
    }

    allowed_confidence = {"low", "medium", "high"}

    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    if result["documented_bound_status"] not in allowed_documented_statuses:
        raise ValueError(
            f"Invalid documented_bound_status: {result['documented_bound_status']}"
        )

    if result["suggested_clip_status"] not in allowed_clip_statuses:
        raise ValueError(
            f"Invalid suggested_clip_status: {result['suggested_clip_status']}"
        )

    if result["confidence"] not in allowed_confidence:
        raise ValueError(f"Invalid confidence: {result['confidence']}")

    # If the model says there is insufficient information, force documented bounds to null.
    if result["documented_bound_status"] == "insufficient_information":
        result["documented_lower_bound"] = None
        result["documented_upper_bound"] = None

    # Suggested clips must be numeric and form a valid interval.
    if result["suggested_lower_clip"] is None:
        raise ValueError("suggested_lower_clip cannot be null")

    if result["suggested_upper_clip"] is None:
        raise ValueError("suggested_upper_clip cannot be null")

    if result["suggested_lower_clip"] >= result["suggested_upper_clip"]:
        raise ValueError(
            f"Invalid clipping interval for {result['feature']}: "
            f"{result['suggested_lower_clip']} >= {result['suggested_upper_clip']}"
        )

    # Conservative confidence calibration.
    if result["documented_bound_status"] in {
        "semantic_public_bound",
        "insufficient_information",
    }:
        if result["confidence"] == "high":
            result["confidence"] = "medium"

    if result["suggested_clip_status"] == "chosen_clipping_bound":
        if result["confidence"] == "high":
            result["confidence"] = "medium"

    return result


def infer_public_bounds_batch(
        tokenizer,
        model,
        model_id: str,
        columns: list[dict[str, str]],
        dataset_context: str,
        max_new_tokens: int = 900,
) -> list[dict[str, Any]]:
    messages = build_batch_messages(
        columns=columns,
        dataset_context=dataset_context,
    )

    text = apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        model_id=model_id,
    )

    print(text)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    raw_results = extract_json_array(response)
    results = [validate_result(r) for r in raw_results]

    expected_names = {c["name"] for c in columns}
    returned_names = {r["feature"] for r in results}

    missing = expected_names - returned_names
    extra = returned_names - expected_names

    if missing:
        raise ValueError(f"Model missed columns: {missing}")

    if extra:
        raise ValueError(f"Model returned unknown columns: {extra}")

    return results


def gemini_contents_from_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    )


def save_prompt_cache(
        dataset_slug: str,
        dataset_name: str,
        columns: list[dict[str, Any]],
        dataset_context: str,
        batch_index: int,
) -> None:
    PROMPTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    messages = build_batch_messages(
        columns=columns,
        dataset_context=dataset_context,
    )
    model_name = GEMINI_MODEL_ID if LLM_PROVIDER == "gemini" else QWEN_MODEL_ID
    safe_model_name = safe_cache_name(model_name)
    path = PROMPTS_CACHE_DIR / (
        f"{safe_cache_name(dataset_slug)}_{LLM_PROVIDER}_{safe_model_name}"
        f"_batch_{batch_index:03d}.txt"
    )
    with path.open("w", encoding="utf-8") as prompt_file:
        prompt_file.write(gemini_contents_from_messages(messages))
        prompt_file.write("\n")
    print(f"Saved LLM prompt cache to {path}", flush=True)


def build_gemini_response_schema(types):
    nullable_number = types.Schema(
        type=types.Type.NUMBER,
        nullable=True,
    )

    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            required=[
                "feature",
                "unit",
                "documented_lower_bound",
                "documented_upper_bound",
                "documented_bound_status",
                "suggested_lower_clip",
                "suggested_upper_clip",
                "suggested_clip_status",
                "evidence",
                "guess_rationale",
                "confidence",
            ],
            properties={
                "feature": types.Schema(type=types.Type.STRING),
                "unit": types.Schema(
                    type=types.Type.STRING,
                    nullable=True,
                ),
                "documented_lower_bound": nullable_number,
                "documented_upper_bound": nullable_number,
                "documented_bound_status": types.Schema(
                    type=types.Type.STRING,
                    enum=[
                        "exact_public_bound",
                        "documented_special_value",
                        "semantic_public_bound",
                        "insufficient_information",
                    ],
                ),
                "suggested_lower_clip": types.Schema(type=types.Type.NUMBER),
                "suggested_upper_clip": types.Schema(type=types.Type.NUMBER),
                "suggested_clip_status": types.Schema(
                    type=types.Type.STRING,
                    enum=[
                        "exact_public_bound",
                        "semantic_public_bound",
                        "chosen_clipping_bound",
                    ],
                ),
                "evidence": types.Schema(type=types.Type.STRING),
                "guess_rationale": types.Schema(type=types.Type.STRING),
                "confidence": types.Schema(
                    type=types.Type.STRING,
                    enum=["low", "medium", "high"],
                ),
            },
        ),
    )


def infer_public_bounds_batch_gemini(
        client,
        model_id: str,
        columns: list[dict[str, str]],
        dataset_context: str,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
) -> list[dict[str, Any]]:
    messages = build_batch_messages(
        columns=columns,
        dataset_context=dataset_context,
    )

    try:
        from google.genai import types
    except ImportError as error:
        raise ImportError(
            "Gemini support requires the google-genai package. "
            "Install it with `pip install google-genai`."
        ) from error

    response = client.models.generate_content(
        model=model_id,
        contents=gemini_contents_from_messages(messages),
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            top_k=GEMINI_TOP_K,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=build_gemini_response_schema(types),
        ),
    )

    raw_results = extract_gemini_json_array(response)
    results = [validate_result(r) for r in raw_results]

    expected_names = {c["name"] for c in columns}
    returned_names = {r["feature"] for r in results}

    missing = expected_names - returned_names
    extra = returned_names - expected_names

    if missing:
        raise ValueError(f"Model missed columns: {missing}")

    if extra:
        raise ValueError(f"Model returned unknown columns: {extra}")

    return results


def infer_public_bounds_all_columns_batched(
        tokenizer,
        model,
        model_id: str,
        columns: list[dict[str, str]],
        dataset_context: str,
        chunk_size: int = 100,
) -> dict[str, dict[str, Any]]:
    all_results: dict[str, dict[str, Any]] = {}

    for batch in chunks(columns, chunk_size):
        try:
            batch_results = infer_public_bounds_batch(
                tokenizer=tokenizer,
                model=model,
                model_id=model_id,
                columns=batch,
                dataset_context=dataset_context,
            )

            for result in batch_results:
                all_results[result["feature"]] = result

        except Exception as batch_error:
            # Fallback: if the batch fails, retry one column at a time.
            for col in batch:
                try:
                    single_result = infer_public_bounds_batch(
                        tokenizer=tokenizer,
                        model=model,
                        model_id=model_id,
                        columns=[col],
                        dataset_context=dataset_context,
                        max_new_tokens=450,
                    )[0]

                    all_results[col["name"]] = single_result

                except Exception as single_error:
                    all_results[col["name"]] = {
                        "feature": col["name"],
                        "unit": None,
                        "documented_lower_bound": None,
                        "documented_upper_bound": None,
                        "documented_bound_status": "error",
                        "suggested_lower_clip": None,
                        "suggested_upper_clip": None,
                        "suggested_clip_status": "error",
                        "evidence": (
                            f"Batch error: {batch_error}; "
                            f"single-column error: {single_error}"
                        ),
                        "guess_rationale": "",
                        "confidence": "low",
                    }

    return all_results


def infer_public_bounds_all_columns_batched_gemini(
        client,
        model_id: str,
        columns: list[dict[str, str]],
        dataset_context: str,
        chunk_size: int = 100,
) -> dict[str, dict[str, Any]]:
    all_results: dict[str, dict[str, Any]] = {}

    for batch in chunks(columns, chunk_size):
        try:
            batch_results = infer_public_bounds_batch_gemini(
                client=client,
                model_id=model_id,
                columns=batch,
                dataset_context=dataset_context,
            )

            for result in batch_results:
                all_results[result["feature"]] = result

        except Exception as batch_error:
            if is_gemini_service_error(batch_error):
                for col in batch:
                    all_results[col["name"]] = error_bound_result(
                        feature=col["name"],
                        error=batch_error,
                    )
                continue

            # Fallback: if the batch fails, retry one column at a time.
            for col in batch:
                try:
                    single_result = infer_public_bounds_batch_gemini(
                        client=client,
                        model_id=model_id,
                        columns=[col],
                        dataset_context=dataset_context,
                        max_output_tokens=2048,
                    )[0]

                    all_results[col["name"]] = single_result

                except Exception as single_error:
                    all_results[col["name"]] = {
                        "feature": col["name"],
                        "unit": None,
                        "documented_lower_bound": None,
                        "documented_upper_bound": None,
                        "documented_bound_status": "error",
                        "suggested_lower_clip": None,
                        "suggested_upper_clip": None,
                        "suggested_clip_status": "error",
                        "evidence": (
                            f"Batch error: {batch_error}; "
                            f"single-column error: {single_error}"
                        ),
                        "guess_rationale": "",
                        "confidence": "low",
                    }

    return all_results


def is_gemini_service_error(error: Exception) -> bool:
    error_text = str(error)
    return (
            "RESOURCE_EXHAUSTED" in error_text
            or "UNAVAILABLE" in error_text
            or "429" in error_text
            or "503" in error_text
    )


def error_bound_result(feature: str, error: Exception) -> dict[str, Any]:
    return {
        "feature": feature,
        "unit": None,
        "documented_lower_bound": None,
        "documented_upper_bound": None,
        "documented_bound_status": "error",
        "suggested_lower_clip": None,
        "suggested_upper_clip": None,
        "suggested_clip_status": "error",
        "evidence": str(error),
        "guess_rationale": "",
        "confidence": "low",
    }


def extract_final_clipping_bounds(
        results: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """
    Convenience helper: extract only the suggested clipping bounds.
    """
    bounds = {}

    for feature, result in results.items():
        lower = result.get("suggested_lower_clip")
        upper = result.get("suggested_upper_clip")

        if lower is not None and upper is not None:
            bounds[feature] = (lower, upper)

    return bounds


def infer_public_bounds_for_columns(
        columns: list[dict[str, str]],
        dataset_context: str,
) -> dict[str, dict[str, Any]]:

    if LLM_PROVIDER == "gemini":
        client = load_gemini_client()
        return infer_public_bounds_all_columns_batched_gemini(
            client=client,
            model_id=GEMINI_MODEL_ID,
            columns=columns,
            dataset_context=dataset_context,
            chunk_size=100,
        )
    elif LLM_PROVIDER == "qwen":
        tokenizer, model = load_qwen_model(
            model_id=QWEN_MODEL_ID,
            local_files_only=False,
        )
        return infer_public_bounds_all_columns_batched(
            tokenizer=tokenizer,
            model=model,
            model_id=QWEN_MODEL_ID,
            columns=columns,
            dataset_context=dataset_context,
            chunk_size=100,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Use 'qwen' or 'gemini'."
        )


def bounds_cache_path(dataset_name: str = "bank_marketing") -> Path:
    model_name = GEMINI_MODEL_ID if LLM_PROVIDER == "gemini" else QWEN_MODEL_ID
    safe_model_name = safe_cache_name(model_name)
    return BOUNDS_CACHE_DIR / f"{dataset_name}_{LLM_PROVIDER}_{safe_model_name}.json"


def load_bounds_cache(
        path: Path,
        required_features: list[str],
) -> dict[str, dict[str, Any]] | None:
    if REFRESH_LLM_BOUNDS or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as cache_file:
        cached = json.load(cache_file)

    results = cached.get("results", cached)
    if not isinstance(results, dict):
        return None

    missing = [
        feature
        for feature in required_features
        if feature not in extract_final_clipping_bounds(results)
    ]
    if missing:
        print(
            f"Ignoring incomplete bounds cache {path}: missing {missing}",
            flush=True,
        )
        return None

    print(f"Loaded cached LLM bounds from {path}", flush=True)
    return results


def save_bounds_cache(
        path: Path,
        results: dict[str, dict[str, Any]],
        columns: list[dict[str, Any]],
        dataset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "provider": LLM_PROVIDER,
        "model": GEMINI_MODEL_ID if LLM_PROVIDER == "gemini" else QWEN_MODEL_ID,
        "features": [column["name"] for column in columns],
        "results": results,
    }
    with path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2)
    print(f"Saved LLM bounds cache to {path}", flush=True)


def bank_marketing_column_metadata() -> list[dict[str, str]]:
    return [
        {
            "name": "age",
            "type": "numeric",
            "description": "Numeric. Age of the bank client, in years.",
        },
        {
            "name": "job",
            "type": "categorical",
            "categories": [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown",
            ],
            "description": (
                "Categorical. Type of job. This feature is ordinal encoded for "
                "preprocessing using public category codes 0..11 for the 12 public "
                "categories: admin., blue-collar, entrepreneur, housemaid, management, "
                "retired, self-employed, services, student, technician, unemployed, unknown."
            ),
        },
        {
            "name": "marital",
            "type": "categorical",
            "categories": ["divorced", "married", "single"],
            "description": (
                "Categorical. Marital status. This feature is ordinal encoded for "
                "preprocessing using public category codes 0..2 for categories: "
                "divorced, married, single."
            ),
        },
        {
            "name": "education",
            "type": "categorical",
            "categories": ["primary", "secondary", "tertiary", "unknown"],
            "description": (
                "Categorical. Education level. This feature is ordinal encoded for "
                "preprocessing using public category codes 0..3 for categories: "
                "primary, secondary, tertiary, unknown."
            ),
        },
        {
            "name": "default",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": (
                "Categorical. Whether the client has credit in default. This feature "
                "is ordinal encoded for preprocessing using public category codes 0..1 "
                "for categories: no, yes."
            ),
        },
        {
            "name": "balance",
            "type": "numeric",
            "description": "Numeric. Average yearly balance, in euros.",
        },
        {
            "name": "housing",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": (
                "Categorical. Whether the client has a housing loan. This feature is "
                "ordinal encoded for preprocessing using public category codes 0..1 "
                "for categories: no, yes."
            ),
        },
        {
            "name": "loan",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": (
                "Categorical. Whether the client has a personal loan. This feature is "
                "ordinal encoded for preprocessing using public category codes 0..1 "
                "for categories: no, yes."
            ),
        },
        {
            "name": "contact",
            "type": "categorical",
            "categories": ["cellular", "telephone", "unknown"],
            "description": (
                "Categorical. Contact communication type. This feature is ordinal "
                "encoded for preprocessing using public category codes 0..2 for "
                "categories: cellular, telephone, unknown."
            ),
        },
        {
            "name": "day",
            "type": "numeric",
            "description": "Numeric. Last contact day of the month.",
        },
        {
            "name": "month",
            "type": "categorical",
            "categories": [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            "description": (
                "Categorical. Last contact month of the year. This feature is ordinal "
                "encoded for preprocessing using public category codes 0..11 for "
                "the 12 calendar months."
            ),
        },
        {
            "name": "duration",
            "type": "numeric",
            "description": "Numeric. Last contact duration, in seconds.",
        },
        {
            "name": "campaign",
            "type": "numeric",
            "description": (
                "Numeric. Number of contacts performed during this campaign and for "
                "this client."
            ),
        },
        {
            "name": "pdays",
            "type": "numeric",
            "description": (
                "Numeric. Number of days that passed after the client was last "
                "contacted from a previous campaign; -1 means client was not "
                "previously contacted."
            ),
        },
        {
            "name": "previous",
            "type": "numeric",
            "description": (
                "Numeric. Number of contacts performed before this campaign and for "
                "this client."
            ),
        },
        {
            "name": "poutcome",
            "type": "categorical",
            "categories": ["failure", "other", "success", "unknown"],
            "description": (
                "Categorical. Outcome of the previous marketing campaign. This "
                "feature is ordinal encoded for preprocessing using public category "
                "codes 0..3 for categories: failure, other, success, unknown."
            ),
        },
    ]


def bank_marketing_dataset_context() -> str:
    return (
        "Bank Marketing dataset from OpenML: direct marketing phone campaigns of a "
        "Portuguese bank. Rows correspond to individual clients or campaign contacts. "
        "Use only public column descriptions. "
        "Numeric guidance: age is a human age in years; balance is an account "
        "balance in euros and can be negative; day is the day of month; duration "
        "is a phone call duration in seconds and cannot be negative; campaign and "
        "previous are counts of contacts and cannot be negative; pdays is days "
        "since previous contact, with -1 explicitly meaning not previously contacted. "
        "Do not use empirical min/max values unless explicitly provided in the public "
        "description. Categorical features are handled outside the LLM call, so all "
        "target columns in this prompt are numeric."
    )


def pima_diabetes_column_metadata() -> list[dict[str, Any]]:
    return [
        {
            "name": "preg",
            "type": "numeric",
            "description": "Numeric. Number of pregnancies.",
            "unit": "count",
        },
        {
            "name": "plas",
            "type": "numeric",
            "description": "Numeric. Plasma glucose concentration from a 2-hour oral glucose tolerance test.",
            "unit": "mg/dL",
        },
        {
            "name": "pres",
            "type": "numeric",
            "description": "Numeric. Diastolic blood pressure.",
            "unit": "mm Hg",
        },
        {
            "name": "skin",
            "type": "numeric",
            "description": "Numeric. Triceps skin fold thickness.",
            "unit": "mm",
        },
        {
            "name": "insu",
            "type": "numeric",
            "description": "Numeric. 2-hour serum insulin.",
            "unit": "mu U/mL",
        },
        {
            "name": "mass",
            "type": "numeric",
            "description": "Numeric. Body mass index.",
            "unit": "kg/m^2",
        },
        {
            "name": "pedi",
            "type": "numeric",
            "description": "Numeric. Diabetes pedigree function.",
            "unit": "score",
        },
        {
            "name": "age",
            "type": "numeric",
            "description": "Numeric. Age in years.",
            "unit": "years",
        },
    ]


def pima_diabetes_dataset_context() -> str:
    return (
        "Pima Indians Diabetes dataset from OpenML/UCI. The task is binary "
        "prediction of diabetes onset from diagnostic measurements. Numeric "
        "guidance: pregnancies, glucose, blood pressure, skin thickness, insulin, "
        "BMI, diabetes pedigree function, and age are non-negative; age is adult "
        "age in years. Use only public medical/column semantics for clipping bounds."
    )


def student_performance_column_metadata() -> list[dict[str, Any]]:
    fixed_scale = "Public UCI Student Performance coding uses fixed integer levels."
    return [
        {
            "name": "age",
            "type": "numeric",
            "description": "Numeric. Student age in years.",
            "unit": "years",
        },
        {
            "name": "sex",
            "type": "categorical",
            "categories": ["F", "M"],
            "description": "Categorical. Student sex: F or M.",
        },
        {
            "name": "address",
            "type": "categorical",
            "categories": ["R", "U"],
            "description": "Categorical. Home address type: rural or urban.",
        },
        {
            "name": "famsize",
            "type": "categorical",
            "categories": ["GT3", "LE3"],
            "description": "Categorical. Family size: greater than 3 or less/equal to 3.",
        },
        {
            "name": "Medu",
            "type": "ordinal",
            "public_bounds": [0, 4],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Mother's education, coded 0 to 4.",
        },
        {
            "name": "Fedu",
            "type": "ordinal",
            "public_bounds": [0, 4],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Father's education, coded 0 to 4.",
        },
        {
            "name": "Mjob",
            "type": "categorical",
            "categories": ["at_home", "health", "other", "services", "teacher"],
            "description": "Categorical. Mother's job.",
        },
        {
            "name": "Fjob",
            "type": "categorical",
            "categories": ["at_home", "health", "other", "services", "teacher"],
            "description": "Categorical. Father's job.",
        },
        {
            "name": "internet",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": "Categorical. Internet access at home.",
        },
        {
            "name": "famsup",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": "Categorical. Family educational support.",
        },
        {
            "name": "paid",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": "Categorical. Extra paid classes within the course subject.",
        },
        {
            "name": "schoolsup",
            "type": "categorical",
            "categories": ["no", "yes"],
            "description": "Categorical. Extra educational school support.",
        },
        {
            "name": "studytime",
            "type": "ordinal",
            "public_bounds": [1, 4],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Weekly study time, coded 1 to 4.",
        },
        {
            "name": "freetime",
            "type": "ordinal",
            "public_bounds": [1, 5],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Free time after school, coded 1 very low to 5 very high.",
        },
        {
            "name": "goout",
            "type": "ordinal",
            "public_bounds": [1, 5],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Going out with friends, coded 1 very low to 5 very high.",
        },
        {
            "name": "Dalc",
            "type": "ordinal",
            "public_bounds": [1, 5],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Workday alcohol consumption, coded 1 very low to 5 very high.",
        },
        {
            "name": "Walc",
            "type": "ordinal",
            "public_bounds": [1, 5],
            "unit": "ordinal level",
            "public_bounds_evidence": fixed_scale,
            "description": "Ordinal. Weekend alcohol consumption, coded 1 very low to 5 very high.",
        },
        {
            "name": "absences",
            "type": "numeric",
            "description": "Numeric. Number of school absences.",
            "unit": "count",
        },
    ]


def student_performance_dataset_context() -> str:
    return (
        "UCI Student Performance Portuguese dataset from OpenML. The task is "
        "pass/fail prediction by thresholding final grade G3 >= 10 as pass. "
        "Columns are demographic, socioeconomic/support, and lifestyle variables. "
        "Categorical and fixed ordinal coded features are handled outside the LLM "
        "call. Numeric guidance: age is student age in years and absences is a "
        "non-negative count."
    )


def compas_column_metadata() -> list[dict[str, Any]]:
    return [
        {
            "name": "sex",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. Binary sex indicator as provided by the OpenML COMPAS dataset.",
        },
        {
            "name": "age",
            "type": "numeric",
            "description": "Numeric. Defendant age in years.",
            "unit": "years",
        },
        {
            "name": "juv_fel_count",
            "type": "numeric",
            "description": "Numeric. Count of juvenile felony charges.",
            "unit": "count",
        },
        {
            "name": "juv_misd_count",
            "type": "numeric",
            "description": "Numeric. Count of juvenile misdemeanor charges.",
            "unit": "count",
        },
        {
            "name": "juv_other_count",
            "type": "numeric",
            "description": "Numeric. Count of other juvenile charges.",
            "unit": "count",
        },
        {
            "name": "priors_count",
            "type": "numeric",
            "description": "Numeric. Count of prior offenses.",
            "unit": "count",
        },
        {
            "name": "age_cat_25-45",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot indicator for age category 25 to 45.",
        },
        {
            "name": "age_cat_Greaterthan45",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot indicator for age greater than 45.",
        },
        {
            "name": "age_cat_Lessthan25",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot indicator for age less than 25.",
        },
        {
            "name": "race_African-American",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot race indicator for African-American.",
        },
        {
            "name": "race_Caucasian",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot race indicator for Caucasian.",
        },
        {
            "name": "c_charge_degree_F",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot indicator for felony charge degree.",
        },
        {
            "name": "c_charge_degree_M",
            "type": "categorical",
            "categories": ["0", "1"],
            "description": "Categorical. One-hot indicator for misdemeanor charge degree.",
        },
    ]


def compas_dataset_context() -> str:
    return (
        "COMPAS two-year recidivism dataset from OpenML. The task is binary "
        "prediction of whether a defendant reoffends within two years. "
        "Categorical one-hot and binary features are handled outside the LLM call. "
        "Numeric guidance: age is adult age in years; juvenile counts and prior "
        "offense counts are non-negative integer counts."
    )


def acs_income_column_metadata() -> list[dict[str, Any]]:
    fixed_code = "ACS PUMS public-use coding defines this as a fixed coded feature."
    return [
        {
            "name": "AGEP",
            "type": "numeric",
            "description": "Numeric. Person age in years.",
            "unit": "years",
        },
        {
            "name": "SEX",
            "type": "coded_categorical",
            "public_bounds": [1, 2],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Sex, ACS codes 1 to 2.",
        },
        {
            "name": "RAC1P",
            "type": "coded_categorical",
            "public_bounds": [1, 9],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Recoded detailed race code, ACS codes 1 to 9.",
        },
        {
            "name": "SCHL",
            "type": "coded_categorical",
            "public_bounds": [1, 24],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Educational attainment, ACS codes 1 to 24.",
        },
        {
            "name": "MAR",
            "type": "coded_categorical",
            "public_bounds": [1, 5],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Marital status, ACS codes 1 to 5.",
        },
        {
            "name": "RELP",
            "type": "coded_categorical",
            "public_bounds": [0, 17],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Relationship or household role, ACS codes 0 to 17.",
        },
        {
            "name": "COW",
            "type": "coded_categorical",
            "public_bounds": [1, 9],
            "unit": "ACS code",
            "public_bounds_evidence": fixed_code,
            "description": "Coded categorical. Class of worker, ACS codes 1 to 9.",
        },
        {
            "name": "OCCP",
            "type": "categorical",
            "categories_from_data": True,
            "description": "Categorical. Occupation code; encoded from fixed occupation categories present in the dataset.",
        },
        {
            "name": "WKHP",
            "type": "numeric",
            "description": "Numeric. Usual hours worked per week.",
            "unit": "hours per week",
        },
        {
            "name": "STATE",
            "type": "categorical",
            "categories_from_data": True,
            "description": "Categorical. State identifier in the HuggingFace ACS income dataset.",
        },
    ]


def acs_income_dataset_context() -> str:
    return (
        "Folktables ACS income dataset from HuggingFace. The task is binary "
        "prediction of whether personal income PINCP is greater than 50,000 USD. "
        "Features follow ACS/PUMS public-use definitions. Coded categorical "
        "features and dataset category sets are handled outside the LLM call. "
        "Numeric guidance: AGEP is age in years for the filtered ACS income "
        "population; WKHP is usual hours worked per week."
    )


def maternal_health_risk_column_metadata() -> list[dict[str, Any]]:
    return [
        {
            "name": "Age",
            "type": "numeric",
            "description": "Numeric. Age in years when a woman is pregnant.",
            "unit": "years",
        },
        {
            "name": "SystolicBP",
            "type": "numeric",
            "description": "Numeric. Upper value of blood pressure during pregnancy.",
            "unit": "mm Hg",
        },
        {
            "name": "DiastolicBP",
            "type": "numeric",
            "description": "Numeric. Lower value of blood pressure during pregnancy.",
            "unit": "mm Hg",
        },
        {
            "name": "BS",
            "type": "numeric",
            "description": "Numeric. Blood glucose level in molar concentration.",
            "unit": "mmol/L",
        },
        {
            "name": "BodyTemp",
            "type": "numeric",
            "description": "Numeric. Body temperature.",
            "unit": "F",
        },
        {
            "name": "HeartRate",
            "type": "numeric",
            "description": "Numeric. Heart rate.",
            "unit": "beats per minute",
        },
    ]


def maternal_health_risk_dataset_context() -> str:
    return (
        "UCI Maternal Health Risk dataset collected from hospitals, clinics, and "
        "maternal health care settings in rural Bangladesh. The task is 3-class "
        "prediction of low risk, mid risk, or high risk during pregnancy from "
        "basic clinical measurements. Numeric guidance: age is maternal age in "
        "years; systolic and diastolic blood pressure are measured in mm Hg; "
        "blood glucose is measured in mmol/L and is non-negative; body "
        "temperature is measured in degrees Fahrenheit; heart rate is beats per "
        "minute and non-negative."
    )


def is_categorical_column(column: dict[str, Any]) -> bool:
    return column.get("type") == "categorical"


def has_direct_public_bounds(column: dict[str, Any]) -> bool:
    return "public_bounds" in column


def direct_public_bound_result(column: dict[str, Any]) -> dict[str, Any]:
    lower_bound, upper_bound = column["public_bounds"]
    unit = column.get("unit")
    evidence = column.get(
        "public_bounds_evidence",
        "Bounds come from the public feature definition.",
    )
    return {
        "feature": column["name"],
        "unit": unit,
        "documented_lower_bound": lower_bound,
        "documented_upper_bound": upper_bound,
        "documented_bound_status": "exact_public_bound",
        "suggested_lower_clip": lower_bound,
        "suggested_upper_clip": upper_bound,
        "suggested_clip_status": "exact_public_bound",
        "evidence": evidence,
        "guess_rationale": "The public feature definition gives fixed bounds.",
        "confidence": "high",
    }


def categorical_bound_result(column: dict[str, Any]) -> dict[str, Any]:
    categories = column.get("categories")
    if not categories:
        raise ValueError(f"Categorical column {column['name']} is missing categories")

    upper_bound = len(categories) - 1
    return {
        "feature": column["name"],
        "unit": "ordinal category code",
        "documented_lower_bound": 0,
        "documented_upper_bound": upper_bound,
        "documented_bound_status": "exact_public_bound",
        "suggested_lower_clip": 0,
        "suggested_upper_clip": upper_bound,
        "suggested_clip_status": "exact_public_bound",
        "evidence": (
            "Bounds come from the public category list used for ordinal encoding."
        ),
        "guess_rationale": (
            f"{len(categories)} public categories are encoded as integer codes "
            f"0..{upper_bound}."
        ),
        "confidence": "high",
    }


def direct_bound_result(column: dict[str, Any]) -> dict[str, Any]:
    if is_categorical_column(column):
        return categorical_bound_result(column)
    if has_direct_public_bounds(column):
        return direct_public_bound_result(column)
    raise ValueError(f"Column {column['name']} has no direct public bound")


def needs_llm_bounds(column: dict[str, Any]) -> bool:
    return not is_categorical_column(column) and not has_direct_public_bounds(column)


def infer_public_bounds_for_dataset(
        dataset_slug: str,
        dataset_name: str,
        columns: list[dict[str, Any]],
        dataset_context: str,
) -> dict[str, dict[str, Any]]:
    direct_results = {
        column["name"]: direct_bound_result(column)
        for column in columns
        if not needs_llm_bounds(column)
    }
    numeric_columns = [
        column
        for column in columns
        if needs_llm_bounds(column)
    ]

    print(
        "Skipping LLM for columns with public categorical/fixed bounds: "
        f"{list(direct_results.keys())}",
        flush=True,
    )
    print(
        "Requesting LLM bounds for numeric columns only: "
        f"{[column['name'] for column in numeric_columns]}",
        flush=True,
    )

    if not numeric_columns:
        return direct_results

    for batch_index, batch in enumerate(chunks(numeric_columns, 100)):
        save_prompt_cache(
            dataset_slug=dataset_slug,
            dataset_name=dataset_name,
            columns=batch,
            dataset_context=dataset_context,
            batch_index=batch_index,
        )

    numeric_feature_names = [column["name"] for column in numeric_columns]
    cache_path = bounds_cache_path(dataset_slug)
    cached_numeric_results = load_bounds_cache(
        path=cache_path,
        required_features=numeric_feature_names,
    )
    if cached_numeric_results is not None:
        numeric_results = {
            feature: cached_numeric_results[feature]
            for feature in numeric_feature_names
        }
        return {**direct_results, **numeric_results}

    numeric_results = infer_public_bounds_for_columns(
        columns=numeric_columns,
        dataset_context=dataset_context,
    )
    missing_numeric_bounds = [
        feature
        for feature in numeric_feature_names
        if feature not in extract_final_clipping_bounds(numeric_results)
    ]
    if not missing_numeric_bounds:
        save_bounds_cache(
            path=cache_path,
            results=numeric_results,
            columns=numeric_columns,
            dataset_name=dataset_name,
        )

    return {**direct_results, **numeric_results}


def infer_bank_marketing_public_bounds(
        columns: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return infer_public_bounds_for_dataset(
        dataset_slug="bank_marketing",
        dataset_name="Bank Marketing OpenML 44234",
        columns=columns,
        dataset_context=bank_marketing_dataset_context(),
    )


def encode_case_study_features(
        x_df,
        columns: list[dict[str, Any]],
) -> np.ndarray:
    x_columns = []
    for column in columns:
        column_name = column["name"]
        series = x_df[column_name]
        if is_categorical_column(column):
            categories = column["categories"]
            category_to_code = {
                category: float(code)
                for code, category in enumerate(categories)
            }
            encoded = series.astype(str).map(category_to_code)
            if encoded.isna().any():
                unknown_values = sorted(set(series[encoded.isna()].astype(str)))
                raise ValueError(
                    f"Column {column_name} has values outside declared categories: "
                    f"{unknown_values}"
                )
            x_columns.append(encoded.to_numpy(dtype=np.float32))
        else:
            if series.isna().any():
                raise ValueError(f"Column {column_name} contains missing numeric values.")
            numeric = series.astype(np.float32)
            x_columns.append(numeric.to_numpy(dtype=np.float32))

    return np.stack(x_columns, axis=1).astype(np.float32)


def with_data_categories(x_df, columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved_columns = []
    for column in columns:
        resolved = dict(column)
        if resolved.get("categories_from_data"):
            values = x_df[resolved["name"]].dropna().astype(str).unique().tolist()
            resolved["categories"] = sorted(values)
        resolved_columns.append(resolved)
    return resolved_columns


def load_openml_case_study_dataset(
        dataset_id: int,
        columns: list[dict[str, Any]],
        dataset_name: str | None = None,
        target_transform=None,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    import openml
    dataset = openml.datasets.get_dataset(dataset_id)
    target_attr = dataset.default_target_attribute
    x_df, y_df, _, _ = dataset.get_data(
        target=target_attr,
        dataset_format="dataframe",
    )

    column_names = [column["name"] for column in columns]
    missing_columns = sorted(set(column_names) - set(x_df.columns))
    if missing_columns:
        raise ValueError(
            f"OpenML dataset {dataset_id} is missing required columns: "
            f"{missing_columns}"
        )

    columns = with_data_categories(x_df[column_names], columns)
    x = encode_case_study_features(x_df[column_names], columns)
    if target_transform is None:
        y = LabelEncoder().fit_transform(y_df).astype(np.int64)
    else:
        y = target_transform(y_df).astype(np.int64)
    return dataset_name or dataset.name, (x, y), columns


def load_uci_case_study_dataset(
        dataset_id: int,
        columns: list[dict[str, Any]],
        dataset_name: str | None = None,
        target_column: str | None = None,
        target_transform=None,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise ImportError(
            "UCI case-study datasets require the ucimlrepo package. "
            "Install it with `pip install ucimlrepo`."
        ) from error

    dataset = fetch_ucirepo(id=dataset_id)
    x_df = dataset.data.features.copy()
    y_df = dataset.data.targets.copy()

    column_names = [column["name"] for column in columns]
    missing_columns = sorted(set(column_names) - set(x_df.columns))
    if missing_columns:
        raise ValueError(
            f"UCI dataset {dataset_id} is missing required columns: {missing_columns}"
        )

    columns = with_data_categories(x_df[column_names], columns)
    x = encode_case_study_features(x_df[column_names], columns)

    if target_column is not None:
        if target_column not in y_df.columns:
            raise ValueError(
                f"UCI dataset {dataset_id} is missing target column {target_column!r}."
            )
        y_source = y_df[target_column]
    elif y_df.shape[1] == 1:
        y_source = y_df.iloc[:, 0]
    else:
        raise ValueError(
            f"UCI dataset {dataset_id} has multiple target columns: {list(y_df.columns)}"
        )

    if target_transform is None:
        y = LabelEncoder().fit_transform(y_source).astype(np.int64)
    else:
        y = target_transform(y_source).astype(np.int64)
    return dataset_name or dataset.metadata.name, (x, y), columns


def load_huggingface_acs_income_dataset(
        max_rows: int = ACS_INCOME_MAX_ROWS,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "ACS income support requires the datasets package. "
            "Install it with `pip install datasets`."
        ) from error

    dataset = load_dataset(
        "birkhoffg/folktables-acs-income",
        split="train",
        cache_dir=HF_DATASETS_CACHE,
    )
    if max_rows > 0 and max_rows < len(dataset):
        dataset = dataset.shuffle(seed=ACS_INCOME_SEED).select(range(max_rows))
    data_df = dataset.to_pandas()

    columns = acs_income_column_metadata()
    column_names = [column["name"] for column in columns]
    missing_columns = sorted(set(column_names + ["PINCP"]) - set(data_df.columns))
    if missing_columns:
        raise ValueError(
            "HuggingFace ACS income dataset is missing required columns: "
            f"{missing_columns}"
        )

    columns = with_data_categories(data_df[column_names], columns)
    x = encode_case_study_features(data_df[column_names], columns)
    target_values = data_df["PINCP"].astype(np.float32).to_numpy()
    unique_targets = set(np.unique(target_values).tolist())
    if unique_targets <= {0.0, 1.0}:
        y = target_values.astype(np.int64)
    else:
        y = (target_values > 50000.0).astype(np.int64)
    dataset_name = f"Folktables ACS income >50K first {len(data_df)} rows"
    return dataset_name, (x, y), columns


def load_bank_marketing_openml_dataset(
        dataset_id: int = BANK_MARKETING_OPENML_ID,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, str]]]:
    import openml
    dataset = openml.datasets.get_dataset(dataset_id)
    target_attr = dataset.default_target_attribute
    x_df, y_df, _, _ = dataset.get_data(
        target=target_attr,
        dataset_format="dataframe",
    )

    columns = bank_marketing_column_metadata()
    column_names = [column["name"] for column in columns]
    x_df = x_df[column_names]

    x_columns = []
    for column in columns:
        column_name = column["name"]
        series = x_df[column_name]
        if is_categorical_column(column):
            categories = column["categories"]
            category_to_code = {
                category: float(code)
                for code, category in enumerate(categories)
            }
            encoded = series.astype(str).map(category_to_code).fillna(-1.0)
            x_columns.append(encoded.to_numpy(dtype=np.float32))
        else:
            if series.isna().any():
                raise ValueError(f"Column {column_name} contains missing numeric values.")
            numeric = series.astype(np.float32)
            x_columns.append(numeric.to_numpy(dtype=np.float32))

    x = np.stack(x_columns, axis=1).astype(np.float32)
    y = LabelEncoder().fit_transform(y_df).astype(np.int64)
    return dataset.name, (x, y), columns


def load_pima_diabetes_openml_dataset(
        dataset_id: int = PIMA_DIABETES_OPENML_ID,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    return load_openml_case_study_dataset(
        dataset_id=dataset_id,
        columns=pima_diabetes_column_metadata(),
        dataset_name="Pima Indians Diabetes",
    )


def load_student_performance_openml_dataset(
        dataset_id: int = STUDENT_PERFORMANCE_OPENML_ID,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    def pass_fail_target(y_df) -> np.ndarray:
        grades = np.asarray(y_df, dtype=np.float32)
        return (grades >= 10.0).astype(np.int64)

    return load_openml_case_study_dataset(
        dataset_id=dataset_id,
        columns=student_performance_column_metadata(),
        dataset_name="UCI Student Performance Portuguese pass/fail",
        target_transform=pass_fail_target,
    )


def load_compas_openml_dataset(
        dataset_id: int = COMPAS_OPENML_ID,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    return load_openml_case_study_dataset(
        dataset_id=dataset_id,
        columns=compas_column_metadata(),
        dataset_name="COMPAS two-year recidivism",
    )


def load_maternal_health_risk_uci_dataset(
        dataset_id: int = MATERNAL_HEALTH_RISK_UCI_ID,
) -> tuple[str, tuple[np.ndarray, np.ndarray], list[dict[str, Any]]]:
    return load_uci_case_study_dataset(
        dataset_id=dataset_id,
        columns=maternal_health_risk_column_metadata(),
        dataset_name="UCI Maternal Health Risk",
        target_column="RiskLevel",
    )


def case_study_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "slug": "bank_marketing",
            "source_url": "https://www.openml.org/search?type=data&status=active&id=44234",
            "context": bank_marketing_dataset_context(),
            "loader": load_bank_marketing_openml_dataset,
        },
        {
            "slug": "pima_diabetes",
            "source_url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
            "context": pima_diabetes_dataset_context(),
            "loader": load_pima_diabetes_openml_dataset,
        },
        {
            "slug": "compas_two_years",
            "source_url": "https://www.kaggle.com/datasets/danofer/compass",
            "context": compas_dataset_context(),
            "loader": load_compas_openml_dataset,
        },
        {
            "slug": "acs_income",
            "source_url": "https://huggingface.co/datasets/birkhoffg/folktables-acs-income",
            "context": acs_income_dataset_context(),
            "loader": load_huggingface_acs_income_dataset,
        },
        {
            "slug": "maternal_health_risk",
            "source_url": "https://archive.ics.uci.edu/dataset/863/maternal+health+risk",
            "context": maternal_health_risk_dataset_context(),
            "loader": load_maternal_health_risk_uci_dataset,
        },
    ]
    selected = os.getenv("CASE_STUDY_DATASETS")
    if selected:
        selected_slugs = {
            slug.strip()
            for slug in selected.split(",")
            if slug.strip()
        }
        specs = [spec for spec in specs if spec["slug"] in selected_slugs]
    return specs


def clipping_bounds_to_arrays(
        columns: list[dict[str, str]],
        results: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    clipping_bounds = extract_final_clipping_bounds(results)

    missing = [
        column["name"]
        for column in columns
        if column["name"] not in clipping_bounds
    ]
    if missing:
        error_details = {
            feature: results[feature].get("evidence")
            for feature in missing
            if feature in results and results[feature].get("documented_bound_status") == "error"
        }
        raise ValueError(
            "Missing clipping bounds for columns: "
            f"{missing}. Categorical columns should be computed directly; "
            "numeric columns must be returned successfully by the LLM. "
            f"LLM error details: {error_details}"
        )

    feature_mins = np.array(
        [clipping_bounds[column["name"]][0] for column in columns],
        dtype=np.float32,
    )
    feature_maxs = np.array(
        [clipping_bounds[column["name"]][1] for column in columns],
        dtype=np.float32,
    )
    return feature_mins, feature_maxs


def llm_inference():
    set_seed(42)

    columns = bank_marketing_column_metadata()
    results = infer_bank_marketing_public_bounds(
        columns=columns,
    )

    print("Full model-assisted bound annotations:")
    print(json.dumps(results, indent=2))

    clipping_bounds = extract_final_clipping_bounds(results)

    print("\nSuggested clipping bounds only:")
    print(json.dumps(clipping_bounds, indent=2))


def empty_metrics_dict():
    return {
        "accs": [],
        "macro_accs": [],
        "probs": [],
        "aucs": [],
        "losses": [],
    }


def compute_oracle_feature_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    return (
        np.nanmin(x, axis=0).astype(np.float32),
        np.nanmax(x, axis=0).astype(np.float32),
    )


def apply_asinh_to_numeric_features(
        x: np.ndarray,
        feature_mins: np.ndarray,
        feature_maxs: np.ndarray,
        columns: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).copy()
    feature_mins = np.asarray(feature_mins, dtype=np.float32).copy()
    feature_maxs = np.asarray(feature_maxs, dtype=np.float32).copy()
    if not END_TO_END_ASINH_NUMERIC:
        return x, feature_mins, feature_maxs

    numeric_mask = np.array(
        [column.get("type") == "numeric" for column in columns],
        dtype=bool,
    )
    x[:, numeric_mask] = np.asinh(x[:, numeric_mask])
    feature_mins[numeric_mask] = np.asinh(feature_mins[numeric_mask])
    feature_maxs[numeric_mask] = np.asinh(feature_maxs[numeric_mask])
    return x, feature_mins, feature_maxs


def evaluate_privtab_on_split(model, train_x, train_y, test_x, test_y,
                             features_count, num_labels, mu, batch_size=1024):
    device = next(model.parameters()).device
    xc = torch.as_tensor(train_x, dtype=torch.float32, device=device).unsqueeze(0)
    yc = torch.as_tensor(train_y, dtype=torch.long, device=device).unsqueeze(0)
    xt = torch.as_tensor(test_x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        summary = model.summarize(xc, yc, mu, d=features_count,
                                  normalize_perturbed_output=True)
        logits = torch.cat([model.predict_from_summary(summary, chunk, d=features_count)
                            for chunk in xt.split(batch_size, dim=1)], dim=1)[0, :, :num_labels]
        probabilities = logits.softmax(-1).cpu().numpy()
        loss = F.cross_entropy(logits, torch.as_tensor(test_y, dtype=torch.long, device=device)).item()
    predicted = probabilities.argmax(-1)
    return (float(np.mean(predicted == test_y)),
            compute_macro_accuracy(test_y, predicted, num_labels), probabilities,
            compute_auc_metric(test_y, probabilities, num_labels), loss)


def append_metrics(
        metrics: dict[str, list[Any]],
        acc_value: float,
        macro_acc_value: float,
        raw_probs: np.ndarray,
        auc_value: float,
        loss_value: float,
) -> None:
    metrics["accs"].append(acc_value)
    metrics.setdefault("macro_accs", []).append(macro_acc_value)
    metrics["probs"].append(raw_probs)
    metrics["aucs"].append(auc_value)
    metrics["losses"].append(loss_value)


def bootstrap_mean_ci(
        values: list[float],
        confidence: float = 0.95,
        rounds: int = 10000,
        seed: int = 0,
) -> tuple[float, float, float]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        return np.nan, np.nan, np.nan
    if values_array.size == 1:
        value = float(values_array[0])
        return value, value, value

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values_array.size,
        size=(rounds, values_array.size),
    )
    bootstrap_means = values_array[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(values_array.mean()), float(lower), float(upper)


def format_bootstrap_ci(values: list[float], scale: float = 1.0) -> str:
    mean, lower, upper = bootstrap_mean_ci(values)
    return f"{mean * scale:.2f} [{lower * scale:.2f}, {upper * scale:.2f}]"


def format_optional_bootstrap_ci(values: list[float] | None, scale: float = 1.0) -> str:
    if not values:
        return "N/A"
    return format_bootstrap_ci(values, scale=scale)


def compute_macro_accuracy(
        y_true: np.ndarray,
        pred_labels: np.ndarray,
        num_labels: int,
) -> float:
    y_true = np.asarray(y_true)
    pred_labels = np.asarray(pred_labels)
    per_class_accs = []
    for label in range(num_labels):
        label_mask = y_true == label
        if not np.any(label_mask):
            continue
        per_class_accs.append(float(np.mean(pred_labels[label_mask] == label)))
    if not per_class_accs:
        return float("nan")
    return float(np.mean(per_class_accs))


def compute_auc_metric(
        y_true: np.ndarray,
        raw_probs: np.ndarray,
        num_labels: int,
) -> float:
    y_true = np.asarray(y_true)
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    try:
        if num_labels == 2:
            if np.unique(y_true).size < 2:
                return 0.5
            return float(roc_auc_score(y_true, raw_probs[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                raw_probs,
                multi_class="ovr",
                average="macro",
                labels=np.arange(num_labels),
            )
        )
    except ValueError:
        return float("nan")


def print_ascii_results_table(
        results_dict: dict[str, Any],
        dataset_name: str,
        mu_values: list[float],
) -> None:
    rows = []
    primary_key = "aucs"
    primary_header = "AUC mean [95% CI]"
    primary_scale = 100.0
    for method_name, method_results in results_dict[dataset_name].items():
        if method_name == "binary":
            continue
        for mu in mu_values:
            if mu not in method_results:
                continue
            metrics = method_results[mu]
            rows.append(
                [
                    method_name,
                    display_mu(method_name, mu),
                    format_bootstrap_ci(metrics["accs"], scale=100.0),
                    format_optional_bootstrap_ci(metrics.get("macro_accs"), scale=100.0),
                    format_bootstrap_ci(metrics[primary_key], scale=primary_scale),
                    format_bootstrap_ci(metrics["losses"], scale=1.0),
                ]
            )

    headers = [
        "Method",
        "mu",
        "ACC mean [95% CI]",
        "Macro ACC mean [95% CI]",
        primary_header,
        "Loss mean [95% CI]",
    ]
    widths = [
        max(len(str(row[index])) for row in rows + [headers])
        for index in range(len(headers))
    ]

    def separator() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: list[str]) -> str:
        return (
                "| "
                + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row)
        )
                + " |"
        )

    print(f"\n{dataset_name}", flush=True)
    print(separator(), flush=True)
    print(format_row(headers), flush=True)
    print(separator(), flush=True)
    for row in rows:
        print(format_row(row), flush=True)
    print(separator(), flush=True)


def display_mu(method_name: str, mu: float) -> str:
    non_dp_reference_methods = {
        "logistic_regression_exact_zscore",
        "gradient_boosting_exact_zscore",
    }
    if method_name in non_dp_reference_methods:
        return "N/A"
    return str(mu)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.6g}"


def bounds_comparison_rows(
        columns: list[dict[str, Any]],
        x: np.ndarray,
        bounds_results: dict[str, dict[str, Any]],
) -> list[list[str]]:
    oracle_feature_mins, oracle_feature_maxs = compute_oracle_feature_bounds(x)
    rows = []
    for index, column in enumerate(columns):
        result = bounds_results[column["name"]]
        is_direct_categorical = not needs_llm_bounds(column)
        display_type = "categorical" if is_direct_categorical else column.get("type", "numeric")
        if display_type != "categorical":
            display_type = "numeric"
        source = "categorical" if is_direct_categorical else "llm"
        confidence = "exact" if is_direct_categorical else str(result.get("confidence", ""))
        rows.append(
            [
                column["name"],
                display_type,
                format_float(float(oracle_feature_mins[index])),
                format_float(float(oracle_feature_maxs[index])),
                source,
                format_float(result.get("suggested_lower_clip")),
                format_float(result.get("suggested_upper_clip")),
                confidence,
            ]
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def metric_summary_rows(
        dataset_results: dict[str, Any],
        mu_values: list[float],
) -> tuple[list[str], list[list[str]]]:
    primary_key = "aucs"
    primary_header = "AUC mean [95% CI]"
    primary_scale = 100.0
    headers = [
        "Method",
        "mu",
        "ACC mean [95% CI]",
        "Macro ACC mean [95% CI]",
        primary_header,
    ]
    rows = []
    for method_name, method_results in dataset_results.items():
        if method_name == "binary":
            continue
        for mu in mu_values:
            if mu not in method_results:
                continue
            metrics = method_results[mu]
            rows.append(
                [
                    method_name,
                    display_mu(method_name, mu),
                    format_bootstrap_ci(metrics["accs"], scale=100.0),
                    format_optional_bootstrap_ci(metrics.get("macro_accs"), scale=100.0),
                    format_bootstrap_ci(metrics[primary_key], scale=primary_scale),
                ]
            )
    return headers, rows


def write_case_study_report(
        path: Path,
        dataset_records: dict[str, dict[str, Any]],
        repeats_dict: dict[str, Any],
        mu_values: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# End-to-End Case Study Report",
        "",
        "This report compares exact empirical data ranges with the public bounds "
        "used by preprocessing. Categorical and fixed public ordinal bounds are "
        "computed directly from public encodings; only remaining numeric features "
        "are sent to the LLM and cached after a complete valid response.",
    ]

    for dataset_name, record in dataset_records.items():
        sections.extend(["", f"## {dataset_name}", ""])
        sections.append("### Bounds")
        sections.append(
            markdown_table(
                [
                    "Feature",
                    "Type",
                    "Exact data min",
                    "Exact data max",
                    "Bound source",
                    "Public/LLM lower",
                    "Public/LLM upper",
                    "Confidence",
                ],
                bounds_comparison_rows(
                    columns=record["columns"],
                    x=record["x"],
                    bounds_results=record["bounds_results"],
                ),
            )
        )
        sections.extend(["", "### Results"])
        headers, rows = metric_summary_rows(repeats_dict[dataset_name], mu_values)
        sections.append(markdown_table(headers, rows))

    with path.open("w", encoding="utf-8") as report_file:
        report_file.write("\n".join(sections) + "\n")
    print(f"Wrote detailed case-study report to {path}", flush=True)


def write_case_study_results_csv(
        path: Path,
        repeats_dict: dict[str, Any],
        mu_values: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "is_binary",
        "method",
        "mu",
        "num_repeats",
        "acc_mean",
        "acc_ci_lower",
        "acc_ci_upper",
        "macro_acc_mean",
        "macro_acc_ci_lower",
        "macro_acc_ci_upper",
        "auc_mean",
        "auc_ci_lower",
        "auc_ci_upper",
        "loss_mean",
        "loss_ci_lower",
        "loss_ci_upper",
    ]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name, dataset_results in repeats_dict.items():
            is_binary = bool(dataset_results.get("binary", False))
            for method_name, method_results in dataset_results.items():
                if method_name == "binary":
                    continue
                for mu in mu_values:
                    if mu not in method_results:
                        continue

                    metrics = method_results[mu]
                    acc_mean, acc_ci_lower, acc_ci_upper = bootstrap_mean_ci(metrics["accs"])
                    macro_values = metrics.get("macro_accs") or []
                    macro_mean, macro_ci_lower, macro_ci_upper = bootstrap_mean_ci(macro_values)
                    auc_mean, auc_ci_lower, auc_ci_upper = bootstrap_mean_ci(metrics["aucs"])
                    loss_mean, loss_ci_lower, loss_ci_upper = bootstrap_mean_ci(metrics["losses"])

                    writer.writerow(
                        {
                            "dataset": dataset_name,
                            "is_binary": is_binary,
                            "method": method_name,
                            "mu": display_mu(method_name, mu),
                            "num_repeats": len(metrics["accs"]),
                            "acc_mean": acc_mean,
                            "acc_ci_lower": acc_ci_lower,
                            "acc_ci_upper": acc_ci_upper,
                            "macro_acc_mean": macro_mean,
                            "macro_acc_ci_lower": macro_ci_lower,
                            "macro_acc_ci_upper": macro_ci_upper,
                            "auc_mean": auc_mean,
                            "auc_ci_lower": auc_ci_lower,
                            "auc_ci_upper": auc_ci_upper,
                            "loss_mean": loss_mean,
                            "loss_ci_lower": loss_ci_lower,
                            "loss_ci_upper": loss_ci_upper,
                        }
                    )

    print(f"Wrote case-study results CSV to {path}", flush=True)


def merge_cached_dp_baseline_results(
        repeats_dict: dict[str, Any],
        results_paths: str = END_TO_END_DP_BASELINES_RESULTS,
) -> None:
    if not results_paths.strip():
        return

    for raw_path in results_paths.split(","):
        stripped_path = raw_path.strip()
        if not stripped_path:
            continue
        path = Path(stripped_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached DP baseline results: {path}")

        cached_results = joblib.load(path)
        if not isinstance(cached_results, dict):
            raise TypeError(
                f"Expected cached results dict in {path}, got {type(cached_results).__name__}."
            )

        for dataset_name, dataset_results in cached_results.items():
            repeats_dict.setdefault(dataset_name, {})
            for method_name, method_results in dataset_results.items():
                if method_name == "binary":
                    continue
                target_method_results = repeats_dict[dataset_name].setdefault(
                    method_name,
                    {},
                )
                for mu, metrics in method_results.items():
                    target_method_results[mu] = metrics

        print(f"Loaded cached DP baseline results from {path}", flush=True)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate PrivTab with public bounds and DP preprocessing.')
    parser.add_argument('--small-weights', default=DEFAULT_SMALL_CONTEXT_WEIGHTS)
    parser.add_argument('--large-weights', default=DEFAULT_LARGE_CONTEXT_WEIGHTS)
    parser.add_argument('--context-switch-threshold', type=int, default=CONTEXT_MODEL_SWITCH_THRESHOLD)
    parser.add_argument('--case-datasets-cache', default=os.getenv('END_TO_END_CASE_DATASETS_CACHE', ''))
    parser.add_argument('--initialize-case-datasets-cache', action='store_true')
    parser.add_argument('--allow-llm-bounds-refresh', action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--dataset-index', type=int)
    parser.add_argument('--indices-dir', default=END_TO_END_INDICES_DIR)
    parser.add_argument('--repeats', type=int, default=int(os.getenv('END_TO_END_REPEATS', '10')))
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--mu-values', type=float, nargs='+', default=[0.4])
    parser.add_argument('--preprocessing-mu-fraction', type=float, default=0.6)
    parser.add_argument('--preprocessing', nargs='+', choices=['exact_zscore', 'dp_oracle_bounds', 'dp_llm_bounds'],
                        default=['exact_zscore', 'dp_oracle_bounds', 'dp_llm_bounds'])
    parser.add_argument('--output', type=Path, default=Path('runs/end_to_end_cases_privtab.pickle'))
    parser.add_argument('--device', default=str(DEVICE))
    parser.add_argument('--batch-size', type=int, default=1024)
    return parser.parse_args()


def main_dp_pre_processing_comparison():
    from types import SimpleNamespace
    from privtab.checkpoints import load_model
    from experiments import run_end_to_end_dp_baselines as workflow

    args = parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        raise ValueError('Repeats and query batch size must be positive.')
    specs = workflow.select_case_studies(case_study_specs(), args.dataset, args.dataset_index)
    case_studies = workflow.load_case_studies_from_cache(args.case_datasets_cache, specs)
    if case_studies is None:
        case_studies = [workflow.load_case_study(spec, args.allow_llm_bounds_refresh) for spec in specs]
    if args.initialize_case_datasets_cache:
        workflow.write_case_studies_cache(args.case_datasets_cache, case_studies)
        return

    models, results, records = {}, {}, {}
    for case in case_studies:
        name, x, y = case['dataset_name'], case['x'], case['y']
        classes = int(np.max(y)) + 1  # Public benchmark class vocabulary.
        results[name] = {'binary': classes == 2}
        if 'columns' in case and 'bounds_results' in case:
            records[name] = case
        workflow.ensure_indices(y, case['slug'], args.indices_dir, args.repeats, args.train_fraction)
        for repeat in range(args.repeats):
            indices = workflow.load_repeat_indices(args.indices_dir, case['slug'], repeat)
            context_size = int(round(len(x) * args.train_fraction))
            weights = args.small_weights if context_size < args.context_switch_threshold else args.large_weights
            if weights not in models:
                models[weights] = load_model(weights, args.device)
            for mu in args.mu_values:
                for mode in args.preprocessing:
                    settings = SimpleNamespace(preprocessing=mode, train_fraction=args.train_fraction,
                                               preprocessing_mu_fraction=args.preprocessing_mu_fraction,
                                               llm_bounds_scaling='dp_zscore')
                    split, model_mu, _ = workflow.preprocess_for_mode(settings, case, indices, mu)
                    xc, yc, xt, yt, features = split
                    values = evaluate_privtab_on_split(models[weights], xc, yc, xt, yt, features,
                                                       classes, model_mu, args.batch_size)
                    metric = results[name].setdefault(f'privtab_{mode}', {}).setdefault(mu, empty_metrics_dict())
                    append_metrics(metric, *values)
                    metric.setdefault('repeat_ids', []).append(repeat)
                    print(f'{name} repeat={repeat} mu={mu:g} {mode}: AUC={values[3]:.4f} loss={values[4]:.4f}', flush=True)
        print_ascii_results_table(results, name, args.mu_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, args.output)
    merge_cached_dp_baseline_results(results)
    write_case_study_results_csv(args.output.with_suffix('.csv'), results, args.mu_values)
    if records:
        write_case_study_report(args.output.with_suffix('.md'), records, results, args.mu_values)
    print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main_dp_pre_processing_comparison()

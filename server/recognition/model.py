"""Portable two-asset CPU inference: one ONNX graph and one rejection JSON.

This module is also copied verbatim to recognition/model.py in a server update.
It deliberately imports no training, corpus, policy, or release modules.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np
import onnxruntime as ort

from .images import PREPROCESSING

LABELS = tuple("அஆஇஈஉஊஎஏஐஒஓஔஃ")
MODEL_LIMIT = 16 * 1024 * 1024
JSON_LIMIT = 2 * 1024 * 1024
TOP_FIELDS = {"schema_version", "format", "model_version", "model", "labels", "preprocessing",
              "rejection", "supportedness", "policy", "provenance", "qualification"}
POLICY_FIELDS = {"schema_version", "policy_version", "model_version", "min_score", "min_similarity", "min_margin"}
RULE_FIELDS = {"min_score", "min_similarity", "min_margin", "min_support"}
COMPONENTS = {"source_release_sha256", "classifier_sha256", "supportedness_sha256", "policy_sha256", "evaluation_sha256"}


def validate_config(value) -> dict:
    """Validate the complete finite JSON contract and return an independent copy."""
    config = _json_copy(value)
    _fields(config, TOP_FIELDS)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["format"] != "tamil-symbols-unified-v1":
        raise ValueError("Unsupported unified model schema.")
    version = config["model_version"]
    _version(version)
    model = config["model"]
    _fields(model, {"sha256", "size"})
    _digest(model["sha256"])
    if type(model["size"]) is not int or not 1 <= model["size"] <= MODEL_LIMIT:
        raise ValueError("Invalid model byte count.")
    if config["labels"] != list(LABELS) or not _exact(config["preprocessing"], PREPROCESSING):
        raise ValueError("Label or preprocessing contract differs from this runtime.")
    rejection = config["rejection"]
    _fields(rejection, {"confidence", "similarity", "prototypes"})
    rejection["confidence"] = _threshold(rejection["confidence"])
    rejection["similarity"] = _threshold(rejection["similarity"])
    prototypes = rejection["prototypes"]
    if type(prototypes) is not list or len(prototypes) != 13:
        raise ValueError("Expected 13 shape prototypes.")
    for row in prototypes:
        if type(row) is not list or len(row) != 576 or any(type(number) not in (int, float) for number in row):
            raise ValueError("Expected numeric 576-dimensional prototypes.")
    _array(prototypes, (13, 576), np.float32)
    support = config["supportedness"]
    _fields(support, {"method", "threshold", "embedding_size"})
    if support["method"] != "frozen-embedding-mlp-v1" or type(support["embedding_size"]) is not int or support["embedding_size"] != 576:
        raise ValueError("Unsupported binary rejection architecture.")
    support["threshold"] = _threshold(support["threshold"])
    config["policy"] = _policy(config["policy"], version, rejection)
    provenance = config["provenance"]
    _fields(provenance, {"source", "attribution", "components", "changes"})
    source = provenance["source"]
    if type(source) is not dict or source.get("license") != "CC-BY-4.0" or source.get("deployment_compatible") is not True:
        raise ValueError("Source permission gate is closed.")
    if type(provenance["attribution"]) is not dict or not provenance["attribution"]:
        raise ValueError("Missing source attribution.")
    _fields(provenance["components"], COMPONENTS)
    for digest in provenance["components"].values():
        _digest(digest)
    _nonblank(provenance["changes"])
    qualification = config["qualification"]
    _fields(qualification, {"qualified", "beta_qualified", "reason"})
    if qualification["qualified"] is not False or qualification["beta_qualified"] is not False:
        raise ValueError("This runtime package must remain explicitly experimental.")
    _nonblank(qualification["reason"])
    return config


def load_json(path: Path) -> dict:
    """Load bounded, regular, non-symlink JSON, rejecting duplicate keys/NaN."""
    try:
        payload = _read_bytes(Path(path), JSON_LIMIT)
        return validate_config(json.loads(payload, object_pairs_hook=_unique_fields,
                                          parse_constant=_invalid_constant))
    except Exception as exc:
        raise ValueError("Invalid unified rejection configuration.") from exc


def load_model(directory: Path, threads: int = 1):
    """Load exactly the two required asset names into one CPU ONNX session.

    Unrelated files are ignored; there is no legacy manifest or model fallback.
    A real warmup validates all three outputs before readiness. Each subsequent
    prediction performs exactly one session.run, covering both learned networks.
    """
    try:
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ValueError("CPU threads must be an integer between 1 and 32.")
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Invalid model directory.")
        config = load_json(directory / "rejection.json")
        model = _read_bytes(directory / "model.onnx", MODEL_LIMIT)
        if len(model) != config["model"]["size"] or hashlib.sha256(model).hexdigest() != config["model"]["sha256"]:
            raise ValueError("Unified ONNX model checksum or size mismatch.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Supplying bytes, not a path, prevents model-relative external weights.
        session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        _signatures(session)
        prototypes = np.asarray(config["rejection"]["prototypes"], dtype=np.float32)

        def predict(tensor):
            _float32_output(tensor, (1, 3, 224, 224))
            logits, embedding, support_logit = session.run(["logits", "embedding", "support_logit"], {"image": tensor})
            character = _decide(logits, embedding, support_logit, config, prototypes)
            return {"status": "recognized" if character is not None else "unrecognized",
                    "character": character, "model_version": config["model_version"]}

        predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
        predict.policy_version = config["policy"]["policy_version"]
        canonical = json.dumps(config["policy"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        predict.policy_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return predict
    except Exception as exc:
        raise ValueError("Invalid unified experimental model package.") from exc


def decide_outputs(logits, embedding, support_logit, config) -> str | None:
    """Pure counterpart of the previous policy plus learned-support decision."""
    checked = validate_config(config)
    prototypes = np.asarray(checked["rejection"]["prototypes"], dtype=np.float32)
    return _decide(logits, embedding, support_logit, checked, prototypes)


def _decide(logits, embedding, support_logit, config, prototypes):
    _float32_output(logits, (1, 13))
    _float32_output(embedding, (1, 576))
    _float32_output(support_logit, (1, 1))
    values = logits.astype(np.float64).reshape(13)
    scores = np.exp(values - values.max())
    scores /= scores.sum()
    index = int(scores.argmax())
    top = np.partition(scores, -2)[-2:]
    margin = float(top.max() - top.min())
    features = embedding.reshape(576)
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(features))
    if not math.isfinite(norm) or norm < 1e-8:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        similarity = float(prototypes[index] @ (features / norm))
    if not math.isfinite(similarity):
        return None
    support_value = float(support_logit[0, 0])
    exponential = math.exp(-support_value if support_value >= 0 else support_value)
    support = 1 / (1 + exponential) if support_value >= 0 else exponential / (1 + exponential)
    policy = config["policy"]
    if _passes(scores[index], similarity, margin, policy) and support >= config["supportedness"]["threshold"]:
        return LABELS[index]
    if policy["schema_version"] == 2:
        rule = policy["character_overrides"].get(LABELS[index])
        if rule is not None and _passes(scores[index], similarity, margin, rule) and support >= rule["min_support"]:
            return LABELS[index]
    return None


def _passes(score, similarity, margin, rule):
    return score >= rule["min_score"] and similarity >= rule["min_similarity"] and margin >= rule["min_margin"]


def _policy(value, version, rejection):
    schema = value.get("schema_version") if type(value) is dict else None
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("Unsupported policy schema.")
    _fields(value, POLICY_FIELDS if schema == 1 else POLICY_FIELDS | {"character_overrides"})
    _version(value["policy_version"])
    _version(value["model_version"])
    if value["model_version"] != version:
        raise ValueError("Policy belongs to another model version.")
    for key in ("min_score", "min_similarity", "min_margin"):
        value[key] = _threshold(value[key])
    if value["min_score"] < rejection["confidence"] or value["min_similarity"] < rejection["similarity"]:
        raise ValueError("Policy cannot weaken embedded calibrated floors.")
    if schema == 2:
        overrides = value["character_overrides"]
        if type(overrides) is not dict or not overrides or set(overrides) - set(LABELS):
            raise ValueError("Invalid predicted-character rule keys.")
        for rule in overrides.values():
            _fields(rule, RULE_FIELDS)
            for key in RULE_FIELDS:
                rule[key] = _threshold(rule[key])
            if rule["min_score"] < rejection["confidence"] or rule["min_similarity"] < max(rejection["similarity"], value["min_similarity"]):
                raise ValueError("Character rule weakens the release score or global shape floor.")
    return value


def _signatures(session):
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)" or not _shape(inputs[0].shape, [3, 224, 224]):
        raise ValueError("Invalid image input signature.")
    outputs = session.get_outputs()
    if len(outputs) != 3 or {output.name for output in outputs} != {"logits", "embedding", "support_logit"}:
        raise ValueError("Invalid unified output names.")
    widths = {"logits": 13, "embedding": 576, "support_logit": 1}
    for output in outputs:
        if output.type != "tensor(float)" or not _shape(output.shape, [widths[output.name]]):
            raise ValueError("Invalid unified output signature.")


def _shape(actual, dimensions):
    return (isinstance(actual, list) and len(actual) == len(dimensions) + 1 and actual[1:] == dimensions
            and (actual[0] is None or isinstance(actual[0], str) or type(actual[0]) is int and actual[0] == 1))


def _read_bytes(path, limit):
    if path.is_symlink():
        raise ValueError("Symlinked model assets are unsupported.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("Expected a bounded regular model asset.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        if not payload or len(payload) > limit:
            raise ValueError("Model asset exceeds its byte limit.")
        return payload
    finally:
        os.close(descriptor)


def _float32_output(value, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Expected a finite float32 tensor with the documented shape.")


def _array(value, shape, dtype):
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(value, dtype=dtype)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid finite numeric matrix.")
        return array
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError("Invalid numeric metadata.") from exc


def _fields(value, expected):
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Unexpected or missing metadata fields.")


def _threshold(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError("Expected a finite threshold between zero and one.")
    return float(value)


def _version(value):
    if type(value) is not str or not 1 <= len(value) <= 128 or not value.isprintable() or value.strip() != value:
        raise ValueError("Invalid model or policy version.")


def _nonblank(value):
    if type(value) is not str or not value.strip():
        raise ValueError("Missing descriptive metadata.")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Invalid SHA-256 metadata.")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON metadata field.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite JSON numbers are unsupported.")


def _json_copy(value, depth=0):
    if depth > 32:
        raise ValueError("Metadata nesting exceeds its limit.")
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite metadata values are unsupported.")
        return value
    if type(value) is list:
        return [_json_copy(item, depth + 1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_copy(item, depth + 1) for key, item in value.items()}
    raise ValueError("Expected ordinary JSON metadata values.")


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_exact(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact(left, right) for left, right in zip(actual, expected))
    return actual == expected

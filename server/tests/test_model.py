"""Portable candidate checks use only assets shipped alongside the API."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[1]


def test_portable_loader_and_default_app_exist():
    assert importlib.util.find_spec("recognition.model") is not None
    assert importlib.util.find_spec("app") is not None


@pytest.fixture
def package(tmp_path):
    target = tmp_path / "model"
    shutil.copytree(SERVER / "model", target)
    return target


def refresh_metadata(directory, name):
    """Test malformed metadata beyond the outer checksum boundary."""
    import hashlib
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = (directory / name).read_bytes()
    manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    source_path = directory / "source-release.json"
    source = json.loads(source_path.read_text())
    source["checksums"][name] = manifest["files"][name]["sha256"]
    source_raw = (json.dumps(source) + "\n").encode()
    source_path.write_bytes(source_raw)
    manifest["files"]["source-release.json"] = {"sha256": hashlib.sha256(source_raw).hexdigest(), "size": len(source_raw)}
    manifest_path.write_text(json.dumps(manifest))


def test_actual_candidate_runs_without_claiming_qualification_or_loading_torch():
    script = """
import json, sys
from pathlib import Path
import numpy as np
from recognition.model import load_model
root=Path('model')
before=json.loads((root/'manifest.json').read_text())
assert before['qualified'] is False
predict=load_model(root)
result=predict(np.zeros((1,3,224,224),np.float32))
assert result['model_version']==before['model_version']
assert result['status'] in ('recognized','unrecognized')
assert result['character'] is None or result['character'] in 'அஆஇஈஉஊஎஏஐஒஓஔஃ'
assert json.loads((root/'manifest.json').read_text())==before
assert 'torch' not in sys.modules
assert not any(name=='tamil_symbols' or name.startswith('tamil_symbols.') for name in sys.modules)
print('portable ONNX candidate inference passed')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=SERVER, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_rejection_keeps_same_probability_and_shape_gates():
    from recognition.model import recognize_outputs
    logits = np.array([15.] + [0.] * 12, np.float32)
    prototypes = np.eye(13, 576, dtype=np.float32)
    assert recognize_outputs(logits, prototypes[0], prototypes, .5, .8) == "அ"
    assert recognize_outputs(logits, prototypes[1], prototypes, .5, .8) is None
    assert recognize_outputs(logits, np.zeros(576, np.float32), prototypes, .5, .8) is None


def test_corrupt_or_missing_artifact_fails_closed(package):
    from recognition.model import load_model
    (package / "model.onnx").write_bytes(b"corrupted model")
    with pytest.raises(ValueError):
        load_model(package)


def test_onnx_decode_failure_is_safe_even_after_checksum_refresh(package):
    from recognition.model import load_model
    (package / "model.onnx").write_bytes(b"not an ONNX graph")
    refresh_metadata(package, "model.onnx")
    with pytest.raises(ValueError, match="Invalid experimental model package"):
        load_model(package)


@pytest.mark.parametrize("name,change", [
    ("labels.json", lambda value: list(reversed(value))),
    ("preprocessing.json", lambda value: {**value, "background_percentile": 95}),
    ("rejection.json", lambda value: {**value, "similarity": float("nan")}),
    ("rejection.json", lambda value: {**value, "prototypes": [[0] * 575] * 13}),
    ("rejection.json", lambda value: {**value, "prototypes": [["0"] * 576] * 13}),
    ("provenance.json", lambda value: {**value, "source": {**value["source"], "deployment_compatible": False}}),
])
def test_invalid_metadata_fails_even_with_refreshed_checksums(package, name, change):
    from recognition.model import load_model
    path = package / name
    path.write_text(json.dumps(change(json.loads(path.read_text()))))
    refresh_metadata(package, name)
    with pytest.raises(ValueError):
        load_model(package)


def test_file_manifest_rejects_unexpected_names_and_symlinks(package, tmp_path):
    from recognition.model import load_model
    path = package / "manifest.json"
    original = json.loads(path.read_text())
    modified = json.loads(path.read_text())
    modified["files"]["../outside"] = {"size": 1, "sha256": "0" * 64}
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError):
        load_model(package)
    path.write_text(json.dumps(original))
    # Replace a copied fixture file only; the distributed candidate stays intact.
    target = package / "labels.json"
    moved = tmp_path / "labels.json"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError):
        load_model(package)


def test_default_app_serves_actual_candidate_and_rejects_missing_credentials(tmp_path):
    from fastapi.testclient import TestClient
    from app import create_app
    token = tmp_path / "service-token"
    token.write_text("portable-test-credential-" * 3)
    with TestClient(create_app(token_file=token)) as client:
        assert client.get("/health/ready").status_code == 401
        assert client.get("/health/ready", headers={"Authorization": "Bearer " + token.read_text()}).status_code == 200
    with TestClient(create_app(token_file=None)) as client:
        assert client.get("/health/ready").status_code == 503

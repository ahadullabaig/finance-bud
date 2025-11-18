import json
from pathlib import Path


def test_canonical_vectors_file_exists_and_valid():
    p = Path(__file__).parent / "canonical_vectors.json"
    assert p.exists(), "canonical_vectors.json must exist"
    data = json.loads(p.read_text())
    assert isinstance(data, list), "canonical_vectors.json must be an array of vectors"
    for v in data:
        assert "id" in v and "input" in v and "expected_violations" in v
        assert isinstance(v["expected_violations"], list)

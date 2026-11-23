# tests/test_docs_parity.py
"""
The English and Spanish docs must not drift apart: each pair must have the same
section structure (same number of headings at each level, code blocks ignored).
La documentación en inglés y español debe tener la misma estructura de secciones.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("README.md", "README.es.md"),
    ("CHALLENGE.md", "DESAFIO.md"),
    ("labs/mlflow_318/README.md", "labs/mlflow_318/README.es.md"),
]


def heading_levels(path: Path) -> list[int]:
    levels, in_code = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        m = re.match(r"^(#{1,6}) ", line)
        if m and not in_code:
            levels.append(len(m.group(1)))
    return levels


@pytest.mark.parametrize("en,es", PAIRS)
def test_same_structure(en, es):
    assert heading_levels(ROOT / en) == heading_levels(ROOT / es), (
        f"{en} and {es} have different sections: update both languages")

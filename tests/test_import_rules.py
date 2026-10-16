"""Import rules from CLAUDE.md, checked on the source code.

- The student (ocgf.models, ocgf.training, ocgf.evaluation) never imports the offline teachers (ocgf.teachers).
- ocgf.datasets and ocgf.geometry stay numpy-only, so they run without torch.
"""
import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBPACKAGES = ["datasets", "geometry", "teachers", "models", "training", "evaluation", "utils"]


def imported_modules(path: Path, root: Path = ROOT) -> set:
    """Absolute module names imported by one source file, with relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = path.relative_to(root).with_suffix("").parts[:-1]      # ocgf/models/x.py -> ('ocgf', 'models')
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                module = ".".join(base + ((node.module,) if node.module else ()))
            else:
                module = node.module
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)   # catches `from ocgf import teachers`
    return names


def violations(subpackages, banned, root: Path = ROOT) -> dict:
    """{file: [banned modules it imports]} over every source file of the given ocgf subpackages."""
    found = {}
    for sub in subpackages:
        for path in sorted((root / "ocgf" / sub).rglob("*.py")):
            hits = sorted(m for m in imported_modules(path, root) if any(m == b or m.startswith(b + ".") for b in banned))
            if hits:
                found[path.relative_to(root).as_posix()] = hits
    return found


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackage_imports(sub):
    importlib.import_module(f"ocgf.{sub}")


def test_student_never_imports_teachers():
    assert violations(["models", "training", "evaluation"], ["ocgf.teachers"]) == {}


def test_numpy_only_packages_do_not_import_torch():
    assert violations(["datasets", "geometry"], ["torch", "torchvision"]) == {}


@pytest.mark.parametrize("source", [
    "from ..teachers import io",
    "from ocgf.teachers.mtrans import model",
    "from ocgf import teachers",
    "import ocgf.teachers.detector2d",
])
def test_checker_catches_teacher_imports(tmp_path, source):
    # the rule is only as good as the checker: a planted violation must be found
    bad = tmp_path / "ocgf" / "models" / "bad.py"
    bad.parent.mkdir(parents=True)
    bad.write_text(source + "\n", encoding="utf-8")
    assert list(violations(["models"], ["ocgf.teachers"], root=tmp_path)) == ["ocgf/models/bad.py"]


def test_checker_ignores_similar_names(tmp_path):
    ok = tmp_path / "ocgf" / "models" / "ok.py"
    ok.parent.mkdir(parents=True)
    ok.write_text("import ocgf.teachers_notes\nfrom ..geometry import bev_grid\n", encoding="utf-8")
    assert violations(["models"], ["ocgf.teachers"], root=tmp_path) == {}

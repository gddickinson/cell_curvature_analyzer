"""The package metadata points at things that exist: the build backend and the console-script entry point."""
import ast
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_build_backend_importable():
    module = META["build-system"]["build-backend"].split(":")[0]
    importlib.import_module(module)


def test_console_script_target_exists_and_is_packaged():
    target = META["project"]["scripts"]["cell-curvature-analyzer"]
    mod, func = target.split(":")
    tree = ast.parse((ROOT / f"{mod}.py").read_text())
    assert func in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert mod in META["tool"]["setuptools"].get("py-modules", [])

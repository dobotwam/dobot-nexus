from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path


def model_imports(component: str) -> list[str]:
    """Find concrete or framework model imports, including relative/from imports."""
    src = Path(__file__).resolve().parents[2] / "src"
    found = []
    for path in (src / "dobot_nexus/modules" / component).rglob("*.py"):
        package = ".".join(path.parent.relative_to(src).parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name("." * node.level + (node.module or ""), package)
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            found.extend(
                f"{path.name}:{node.lineno} {name}"
                for name in names
                if name == "dobot_nexus.models" or name.startswith("dobot_nexus.models.")
            )
    return found

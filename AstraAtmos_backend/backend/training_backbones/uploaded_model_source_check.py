"""Static safety gate for uploaded user models.

An uploaded model is executable Python, so the platform refuses to run source that
imports anything outside ``torch``/``numpy`` or calls a process/file builtin. The
check is applied twice on purpose:

* at **upload/revalidate** time, so the user gets an actionable error; and
* every time the platform is about to **execute** the code (training, or rebuilding
  an uploaded model from a checkpoint),

because an Earth checkpoint can carry source text, and that text must satisfy the
same gate as the file it came from. Keeping the rule in one place means the two
paths can never drift apart.
"""

from __future__ import annotations

import ast
from typing import Any

#: Top-level modules an uploaded model may import.
ALLOWED_IMPORT_ROOTS = frozenset({"torch", "numpy"})

#: Builtins that must never be called directly by an uploaded model.
DISALLOWED_DIRECT_CALLS = frozenset({"open", "eval", "exec", "compile", "__import__"})

#: Attribute calls that would let an uploaded model spawn processes or run commands.
DISALLOWED_ATTRIBUTE_CALLS = frozenset({"system", "popen", "Popen", "run"})


def call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def validate_uploaded_model_ast(tree: ast.AST) -> list[str]:
    """Return the list of safety violations in a parsed uploaded model."""
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Disallowed import: {root}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append("Disallowed import: relative import")
                continue
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Disallowed import: {root or '<unknown>'}")
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if isinstance(node.func, ast.Name) and name in DISALLOWED_DIRECT_CALLS:
                errors.append(f"Disallowed call: {name}")
            elif isinstance(node.func, ast.Attribute) and name in DISALLOWED_ATTRIBUTE_CALLS:
                errors.append(f"Disallowed call: {name}")
    return errors


def validate_uploaded_model_source(source_text: str, filename: str = "<uploaded model>") -> list[str]:
    """Parse and safety-check source text, returning violations (empty means safe)."""
    try:
        tree = ast.parse(source_text, filename=filename)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]
    return validate_uploaded_model_ast(tree)


def source_is_safe(source_text: str, filename: str = "<uploaded model>") -> tuple[bool, list[str]]:
    errors = validate_uploaded_model_source(source_text, filename)
    return (not errors), errors


def assert_source_is_safe(source_text: str, filename: str = "<uploaded model>") -> ast.AST:
    """Return the parsed tree, or raise ``ValueError`` listing the violations."""
    try:
        tree = ast.parse(source_text, filename=filename)
    except SyntaxError as exc:
        raise ValueError(f"Uploaded model source is not valid Python: {exc}") from exc
    errors = validate_uploaded_model_ast(tree)
    if errors:
        raise ValueError("Uploaded model source failed the safety check: " + "; ".join(errors))
    return tree


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "DISALLOWED_ATTRIBUTE_CALLS",
    "DISALLOWED_DIRECT_CALLS",
    "assert_source_is_safe",
    "call_name",
    "source_is_safe",
    "validate_uploaded_model_ast",
    "validate_uploaded_model_source",
]

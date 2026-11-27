"""Safety net for splitting raglib/facade.py into a package: the public RAG API must not change.

``rag_api_snapshot.json`` holds every public RAG method, property and classmethod with its exact
signature, plus the full ``__init__`` signature. Regenerate it only for a deliberate API change:
``python tests/unit/test_rag_api_snapshot.py``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict

SNAPSHOT = Path(__file__).with_name("rag_api_snapshot.json")


def public_api() -> Dict[str, str]:
    from raglib import RAG

    api: Dict[str, str] = {"__init__": str(inspect.signature(RAG.__init__))}
    for name in sorted(dir(RAG)):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(RAG, name)
        if isinstance(attr, property):
            api[name] = "property"
        elif isinstance(attr, classmethod):
            api[name] = "classmethod" + str(inspect.signature(attr.__func__))
        elif isinstance(attr, staticmethod):
            api[name] = "staticmethod" + str(inspect.signature(attr.__func__))
        elif callable(attr):
            api[name] = str(inspect.signature(attr))
        else:
            api[name] = f"attribute: {type(attr).__name__}"
    return api


def test_the_public_rag_api_is_unchanged() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = public_api()
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    assert (missing, added, changed) == ([], [], []), (
        f"RAG API changed: missing={missing} added={added} changed={changed}"
    )


def test_rag_keeps_its_module_name() -> None:
    from raglib import RAG

    assert RAG.__module__ == "raglib.facade"


def test_no_method_is_defined_in_two_classes() -> None:
    """Each name lives in one place: no mixin overrides another, and RAG overrides none of them."""

    from raglib import RAG

    owners: Dict[str, list] = {}
    for cls in RAG.__mro__:
        if cls is object or not cls.__module__.startswith("raglib.facade"):
            continue
        for name, value in vars(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if callable(value) or isinstance(value, (property, classmethod, staticmethod)):
                owners.setdefault(name, []).append(cls.__name__)
    duplicates = {name: classes for name, classes in owners.items() if len(classes) > 1}
    assert duplicates == {}


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(public_api(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT}")

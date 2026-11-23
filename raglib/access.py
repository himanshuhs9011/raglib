"""Document-level access control: tag chunks at ingest, build a permission filter per query.

Deny by default. With access control on, a chunk is retrievable only when it carries access
tags *and* the asking user is cleared for all of them:

* ``access_level`` - one of the ordered ``AccessControl.levels`` (default
  ``public < internal < confidential < executive``); the user's ``max_access_level`` must be at
  or above it. A query without a user context is cleared for the lowest level only.
* ``department`` - one name or a list; when set, the user must belong to one of them.
* ``allowed_roles`` - one role or a list; when set, the user must hold one of them.

Tags are stored on every chunk as flat boolean metadata keys (``_acl_*``), so the same filter
is pushed down into every backend - in-memory, Chroma, Qdrant, pgvector and BM25 - as a
pre-filter inside the search, never as a post-filter over results the user may not see. A chunk
without tags lacks those keys and therefore never matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from raglib.vectorstores.base_store import MetadataFilter

DEFAULT_ACCESS_LEVELS: Tuple[str, ...] = ("public", "internal", "confidential", "executive")

ACCESS_LEVEL_KEY = "access_level"
DEPARTMENT_KEY = "department"
ALLOWED_ROLES_KEY = "allowed_roles"
ACCESS_TAG_KEYS: Tuple[str, ...] = (ACCESS_LEVEL_KEY, DEPARTMENT_KEY, ALLOWED_ROLES_KEY)

ACL_PREFIX = "_acl_"
ACL_ANY_DEPARTMENT = "_acl_dept_any"
ACL_ANY_ROLE = "_acl_role_any"


class FilterNotSupportedError(RuntimeError):
    """A retriever or backend was asked to apply a metadata filter it cannot enforce.

    Raised instead of silently returning unfiltered results, which could leak documents.
    """


@dataclass(frozen=True)
class UserContext:
    """Who is asking: the identity a permission filter is built for.

    ``max_access_level`` is the highest ``access_level`` this user may read (``None`` = the
    lowest level only). ``roles`` and ``departments`` are matched against a document's
    ``allowed_roles`` and ``department`` tags.
    """

    user_id: Optional[str] = None
    roles: Sequence[str] = field(default_factory=tuple)
    departments: Sequence[str] = field(default_factory=tuple)
    max_access_level: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize ``roles`` / ``departments`` (a single string is one value) to tuples."""

        object.__setattr__(self, "roles", _as_tuple(self.roles, "roles"))
        object.__setattr__(self, "departments", _as_tuple(self.departments, "departments"))

    @classmethod
    def coerce(cls, value: Union["UserContext", Mapping[str, Any], None]) -> Optional["UserContext"]:
        """Accept a ``UserContext``, a plain dict with the same keys, or ``None``."""

        if value is None or isinstance(value, UserContext):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"user_id", "roles", "departments", "max_access_level"}
            if unknown:
                raise ValueError(f"Unknown user context keys: {sorted(unknown)}")
            return cls(**dict(value))
        raise TypeError(f"user must be a UserContext or a dict, got {type(value).__name__}")


@dataclass(frozen=True)
class AccessControl:
    """Access-control policy for a ``RAG`` (``RAG(..., access_control=AccessControl(...))``).

    ``levels`` orders the access levels from least to most restricted. ``default_access_level``
    is stamped on documents ingested without an ``access_level``; leave it ``None`` (the default)
    to keep untagged documents invisible to every query.
    """

    levels: Sequence[str] = DEFAULT_ACCESS_LEVELS
    default_access_level: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the level list and the default level."""

        levels = _as_tuple(self.levels, "levels")
        if not levels:
            raise ValueError("AccessControl.levels must not be empty")
        if len(set(levels)) != len(levels):
            raise ValueError("AccessControl.levels must be unique")
        object.__setattr__(self, "levels", levels)
        if self.default_access_level is not None:
            self._rank(self.default_access_level)

    @classmethod
    def coerce(cls, value: Union["AccessControl", bool, Mapping[str, Any], None]) -> Optional["AccessControl"]:
        """Accept ``True`` (default policy), an ``AccessControl``, a dict of its fields, or ``None``/``False``."""

        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, AccessControl):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"access_control must be True, a dict or an AccessControl, got {type(value).__name__}")

    def _rank(self, level: str) -> int:
        """Index of ``level`` in ``levels`` (ValueError for an unknown level)."""

        try:
            return list(self.levels).index(level)
        except ValueError:
            raise ValueError(f"Unknown access_level {level!r}. Known levels: {list(self.levels)}") from None

    def tag(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        """Return ``metadata`` with the flat ``_acl_*`` keys for its access tags (stale ones removed).

        Without an ``access_level`` (and no ``default_access_level``) no ``_acl_*`` key is added,
        so the chunk matches no permission filter: deny by default.
        """

        tagged = {key: value for key, value in metadata.items() if not str(key).startswith(ACL_PREFIX)}
        level = tagged.get(ACCESS_LEVEL_KEY)
        if level in (None, "") and self.default_access_level is not None:
            level = self.default_access_level
            tagged[ACCESS_LEVEL_KEY] = level
        if level in (None, ""):
            return tagged
        self._rank(str(level))
        departments = _as_tuple(tagged.get(DEPARTMENT_KEY), DEPARTMENT_KEY)
        roles = _as_tuple(tagged.get(ALLOWED_ROLES_KEY), ALLOWED_ROLES_KEY)
        if departments:
            tagged.update({_acl_key("dept", name): True for name in departments})
        else:
            tagged[ACL_ANY_DEPARTMENT] = True
        if roles:
            tagged.update({_acl_key("role", name): True for name in roles})
        else:
            tagged[ACL_ANY_ROLE] = True
        return tagged

    def filter_for(self, user: Optional[UserContext]) -> MetadataFilter:
        """Build the permission filter for ``user`` (``None`` = anonymous: lowest level, no roles)."""

        max_level = user.max_access_level if user is not None and user.max_access_level else self.levels[0]
        allowed_levels = list(self.levels[: self._rank(max_level) + 1])
        departments = user.departments if user is not None else ()
        roles = user.roles if user is not None else ()
        return {
            "$and": [
                {ACCESS_LEVEL_KEY: {"$in": allowed_levels}},
                {"$or": [{ACL_ANY_DEPARTMENT: True}] + [{_acl_key("dept", name): True} for name in departments]},
                {"$or": [{ACL_ANY_ROLE: True}] + [{_acl_key("role", name): True} for name in roles]},
            ]
        }

    @staticmethod
    def signature(metadata: Mapping[str, Any]) -> str:
        """Stable digest of a chunk's access tags (detects permission-only changes on re-ingest)."""

        tags = {key: metadata[key] for key in sorted(metadata) if key in ACCESS_TAG_KEYS or str(key).startswith(ACL_PREFIX)}
        return hashlib.sha256(json.dumps(tags, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def merge_filters(*filters: Optional[MetadataFilter]) -> Optional[MetadataFilter]:
    """AND together the non-empty filters (a caller filter can only narrow a permission filter)."""

    present = [item for item in filters if item]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {"$and": present}


def access_tags(
    access_level: Optional[str] = None,
    department: Union[str, Sequence[str], None] = None,
    allowed_roles: Union[str, Sequence[str], None] = None,
) -> Dict[str, Any]:
    """Build the metadata dict for ``RAG.add(..., access=...)`` (only the tags that are set)."""

    tags: Dict[str, Any] = {}
    if access_level is not None:
        tags[ACCESS_LEVEL_KEY] = access_level
    if department is not None:
        tags[DEPARTMENT_KEY] = department if isinstance(department, str) else list(department)
    if allowed_roles is not None:
        tags[ALLOWED_ROLES_KEY] = [allowed_roles] if isinstance(allowed_roles, str) else list(allowed_roles)
    return tags


def _acl_key(kind: str, name: str) -> str:
    """Flat metadata key for one department/role (hashed: safe in every backend's key syntax)."""

    return f"{ACL_PREFIX}{kind}_{hashlib.sha256(str(name).encode('utf-8')).hexdigest()[:16]}"


def _as_tuple(value: Any, label: str) -> Tuple[str, ...]:
    """Normalize ``None`` / a string / a sequence of strings to a tuple of non-empty strings."""

    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item))
    raise TypeError(f"{label} must be a string or a list of strings, got {type(value).__name__}")


__all__: List[str] = [
    "AccessControl",
    "UserContext",
    "FilterNotSupportedError",
    "merge_filters",
    "access_tags",
    "DEFAULT_ACCESS_LEVELS",
]

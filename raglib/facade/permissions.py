"""Access control on the facade: the permission filter for a user, scoping filters and memory per user, and access tags on chunks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple, Union

from raglib.access import AccessControl, UserContext, merge_filters
from raglib.schemas import Document
from raglib.vectorstores.base_store import MetadataFilter

_UNTAGGED_SIGNATURE = AccessControl.signature({})


class PermissionsMixin:
    """Access control on the facade: the permission filter for a user, scoping filters and memory per user, and access tags on chunks (mixin of ``RAG``)."""

    @property
    def access_control(self) -> Optional[AccessControl]:
        """The access-control policy (``None`` when access control is off)."""

        return self._access

    def permission_filter(self, user: Union[UserContext, Mapping[str, Any], None] = None) -> Optional[MetadataFilter]:
        """The metadata filter access control applies for ``user`` (``None`` when it is off)."""

        if self._access is None:
            return None
        return self._access.filter_for(UserContext.coerce(user))

    def _scope(
        self,
        filters: Optional[MetadataFilter],
        user: Union[UserContext, Mapping[str, Any], None],
        session_id: Optional[str],
    ) -> Tuple[Optional[MetadataFilter], Optional[str]]:
        """Merge the permission filter into ``filters`` and namespace the memory session by user."""

        context = UserContext.coerce(user)
        if self._access is None:
            if context is not None:
                raise ValueError("user=... needs access_control (RAG(..., access_control=True))")
            return filters, session_id
        acl = self._access.filter_for(context)
        # Memory is per user: two users never read each other's history, even with the same (or no)
        # session_id. Without a user_id the owner is the permission scope itself.
        signature = hashlib.sha256(json.dumps(acl, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if context is not None and context.user_id:
            # User plus their current permissions: a demoted user starts a new history instead of
            # reading answers that quote documents above their new level.
            owner = f"user:{context.user_id}|acl:{signature}"
        else:
            owner = "acl:" + signature
        return merge_filters(acl, filters), f"{owner}::{session_id or 'default'}"

    def _private_document(self, document: Document) -> bool:
        """True for a document with any access restriction: a level above the lowest, or a department
        or role restriction (web results have no access tags)."""

        if self._access is None:
            return False
        metadata = document.metadata
        level = metadata.get("access_level")
        restricted_level = level is not None and level != self._access.levels[0]
        return bool(restricted_level or metadata.get("department") or metadata.get("allowed_roles"))

    def _tag(self, document: Document) -> Document:
        """Apply access-control tags to a chunk or section (no-op without access control)."""

        return replace(document, metadata=self._access.tag(document.metadata)) if self._access is not None else document

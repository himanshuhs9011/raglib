"""Guardrail follow-ups: PII look-alikes and reversible masking, incremental streaming guards,
rate-limit identity and stores, the guard hash in the answer-cache key, the production cost cap,
and no unmasked input in errors, logs or traces (offline, no network)."""

from __future__ import annotations

import json
import logging
from typing import List

import pytest

from raglib import RAG
from raglib.cache import AnswerCache
from raglib.core.base import GuardBlockedError
from raglib.guards import (
    CitationAccessGuard,
    Decision,
    Guard,
    GuardContext,
    GuardStore,
    InMemoryGuardStore,
    MaxOutputTokens,
    MaxQuestionTokens,
    NothingRelevant,
    OutputLeakGuard,
    PIIMasker,
    StreamingOutput,
    UserDailyBudget,
    UserRateLimit,
    allow,
)
from raglib.guards.pii import (
    PlaceholderRestorer,
    find,
    luhn_valid,
    mask_reversible,
    restore,
    verhoeff_valid,
)
from raglib.llm.mock_client import MockLLMClient
from raglib.presets import resolve_preset
from raglib.schemas import Document

CORPUS = [
    "Acme Corp offices close on Friday afternoons in summer and the cafeteria opens at eight.",
    "The Orion launch window opens in March and closes in April for the whole launch team.",
]
EMAIL, CARD = "jane.doe@example.com", "4111 1111 1111 1111"


def _verhoeff_check_digit(number: str) -> str:
    for digit in "0123456789":
        if verhoeff_valid(number + digit):
            return digit
    raise AssertionError


VALID_AADHAAR = "23456789012" + _verhoeff_check_digit("23456789012")
INVALID_AADHAAR = VALID_AADHAAR[:-1] + str((int(VALID_AADHAAR[-1]) + 1) % 10)


class WordStreamer(MockLLMClient):
    """The offline mock, streamed word by word; ``log`` records when each token was produced."""

    def __init__(self, text: str = "", piece: int = 0) -> None:
        self.text, self.piece, self.log, self.prompts = text, piece, [], []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.text or super().complete(prompt, system)

    def stream(self, prompt: str, system: str = ""):
        answer = self.complete(prompt, system)
        if self.piece:
            tokens = [answer[i:i + self.piece] for i in range(0, len(answer), self.piece)]
        else:
            tokens = [word + " " for word in answer.split(" ")]
        for index, token in enumerate(tokens):
            self.log.append(("llm", index))
            yield token


def _rag(llm=None, **kwargs) -> RAG:
    kwargs.setdefault("token_counter", "units")
    return RAG(CORPUS, chat_llm=llm or WordStreamer(), embedding_llm="mock", llm_cache=False, **kwargs)


# ------------------------------------------------------------------ 1. PII: formats and look-alikes


@pytest.mark.parametrize(
    "text",
    [
        "order ORD-2026-4111111111111111 shipped",
        "order #A4111111111111111",
        f"invoice {INVALID_AADHAAR} is due",
        f"invoice {INVALID_AADHAAR[:4]} {INVALID_AADHAAR[4:8]} {INVALID_AADHAAR[8:]} is due",
        "dated 2026-11-23, 23/11/2026 and 2026.11.23",
        "price $4,111.11, ₹1,23,456.78 or 4111111111111111.00",
        "version v2.10.3, build 10.0.26300, release 1.2.3.4",
        "card-like 1234 5678 9012 3456 fails Luhn",
        "mixed separators 4111 1111-1111 1111",
        "PAN-like ABCXE1234F (bad holder type), abcpe1234f (lower case), XABCPE1234F (inside a word)",
        "call 9876543210 or +91 98765 43210",  # phones are opt-in
        "user@localhost and v1.2@3.4",
    ],
)
def test_default_pii_set_ignores_look_alikes(text: str) -> None:
    assert not luhn_valid("1234567890123456") and not verhoeff_valid(INVALID_AADHAAR)
    assert PIIMasker().check(text, GuardContext()).action == "allow", find(text)


def test_default_pii_set_masks_real_values_with_checksums_and_boundaries() -> None:
    text = f"mail {EMAIL}, card {CARD}. Aadhaar {VALID_AADHAAR[:4]} {VALID_AADHAAR[4:8]} {VALID_AADHAAR[8:]}, PAN ABCPE1234F."
    decision = PIIMasker().check(text, GuardContext())
    assert decision.value == "mail <EMAIL_1>, card <CARD_1>. Aadhaar <AADHAAR_1>, PAN <PAN_1>."
    phones = PIIMasker(phone=True).check("call +91 98765 43210 on 2026-11-23 re invoice 123456789012", GuardContext())
    assert phones.value == "call <PHONE_1> on 2026-11-23 re invoice 123456789012"


def test_reversible_masking_is_consistent_and_never_reuses_typed_placeholders() -> None:
    masked, mapping, counts = mask_reversible(f"{EMAIL} wrote to bob@example.org, cc {EMAIL}; see <EMAIL_1>")
    assert masked == "<EMAIL_2> wrote to <EMAIL_3>, cc <EMAIL_2>; see <EMAIL_1>"
    assert mapping == {"<EMAIL_2>": EMAIL, "<EMAIL_3>": "bob@example.org"} and counts == {"EMAIL": 3}
    assert restore(masked, mapping) == f"{EMAIL} wrote to bob@example.org, cc {EMAIL}; see <EMAIL_1>"
    restorer = PlaceholderRestorer(mapping)
    pieces = [restorer.feed(t) for t in ["Hi <EMA", "IL_2>", ", bye <", "EMAIL_3> ok"]] + [restorer.flush()]
    assert "".join(pieces) == f"Hi {EMAIL}, bye bob@example.org ok" and all("<EMA" not in p for p in pieces)


def test_values_are_restored_for_the_caller_only(tmp_path) -> None:
    llm, events = WordStreamer(), []
    rag = RAG(CORPUS, chat_llm=llm, embedding_llm="mock", token_counter="units", memory=True, answer_cache=True,
              input_guards=["pii"], on_event=events.append, output_dir=str(tmp_path))
    question = f"When do Acme offices close? Reply to {EMAIL}, card {CARD}"
    result = rag.query(question, session_id="s")
    assert EMAIL in result.answer and CARD in result.answer  # the mock echoes the question: restored
    streamed = "".join(rag.stream(question, session_id="t"))
    assert EMAIL in streamed and "<EMAIL_1>" not in streamed
    stored = [
        "\n".join(llm.prompts), repr(events), json.dumps(result.metadata, default=str), repr(result.trace_events),
        rag._conversations.get("s").get_context(), repr(vars(rag._answer_cache._cache)),
        *(path.read_text(encoding="utf-8") for path in tmp_path.iterdir()),
    ]
    assert all(EMAIL not in text and "4111 1111" not in text for text in stored)
    assert "<EMAIL_1>" in rag._conversations.get("s").get_context()
    assert all("pii_map" not in json.dumps(d) for d in result.metadata["guards"])
    no_restore = _rag(input_guards=[PIIMasker(restore=False)])
    assert EMAIL not in no_restore.query(question).answer


def test_pii_masker_runs_before_every_other_input_guard() -> None:
    seen: List[str] = []

    class Spy(Guard):
        name = "spy"

        def check(self, value, context):
            seen.extend([value, context.question])
            return allow()

    rag = _rag(input_guards=[Spy(), PIIMasker()])  # listed first, still runs after masking
    rag.query(f"When do Acme offices close? {EMAIL}")
    assert seen and all(EMAIL not in text for text in seen) and "<EMAIL_1>" in seen[0]


# ------------------------------------------------------------------ 2. streaming guards


def test_production_preset_keeps_streaming_with_its_output_guards() -> None:
    llm = WordStreamer()
    rag = RAG(CORPUS, preset="production", chat_llm=llm, embedding_llm="mock", reranker="none", llm_cache=False)
    order: List[str] = []
    for piece in rag.stream("When do Acme offices close on Fridays in summer?"):
        order.append(f"out:{len(llm.log)}")
    produced = len(llm.log)
    assert len(order) > 2
    assert int(order[0].split(":")[1]) < produced  # the first piece left before the model finished


def test_secret_split_across_tokens_is_still_masked() -> None:
    text = "Here is the key sk-live-abcdefghijklmnopqrstuvwxyz0123 for the Acme office. Card " + CARD + " too."
    rag = _rag(WordStreamer(text, piece=3), output_guards=[OutputLeakGuard()])
    pieces = list(rag.stream("When do Acme offices close?"))
    joined = "".join(pieces)
    assert len(pieces) >= 2
    assert "sk-live" not in joined and "4111" not in joined
    assert joined == "Here is the key [API_KEY] for the Acme office. Card [CARD] too."


def test_long_sentences_flow_with_a_holdback_window() -> None:
    monitor = StreamingOutput([OutputLeakGuard()], GuardContext(stage="output"), holdback=40)
    words = [f"word{i} " for i in range(60)]  # no sentence end at all
    early = [monitor.feed(word) for word in words]
    assert sum(1 for piece in early if piece) > 5  # released while the "sentence" is still going
    assert "".join(early) + monitor.close() == "".join(words)


def test_max_output_tokens_truncates_while_streaming_and_stops_reading_the_model() -> None:
    long = " ".join(f"Sentence {i} is here." for i in range(40))
    llm = WordStreamer(long)
    rag = _rag(llm, output_guards=[MaxOutputTokens(max_tokens=12, token_counter=lambda t: len(t.split()))])
    pieces = list(rag.stream("When do Acme offices close?"))
    joined = "".join(pieces)
    assert joined.endswith(" …") and len(joined.split()) <= 14
    assert len(llm.log) < len(long.split())  # the model's stream was closed early


def test_citation_block_mid_stream_stops_and_appends_the_message() -> None:
    public = Document(id="p", content="a", metadata={"access_level": "public"})
    secret = Document(id="s", content="b", metadata={"access_level": "confidential"})
    context = GuardContext(stage="output", documents=[public, secret], permission_filter={"access_level": {"$in": ["public"]}})
    monitor = StreamingOutput([CitationAccessGuard()], context)
    shown = "".join(monitor.feed(t) for t in ["Fact one [1]. ", "Secret fact [2]. ", "More text."]) + monitor.close()
    assert shown.startswith("Fact one [1]. ") and "Secret fact" not in shown
    assert shown.endswith(CitationAccessGuard.block_message) and monitor.stopped and monitor.blocked is not None


def test_streamed_guard_decisions_are_aggregated_and_traced() -> None:
    text = "Key one sk-live-abcdefghijklmnopqrstuvwxyz0123. Key two sk-live-zyxwvutsrqponmlkjihgfedcba9876. Done."
    rag = _rag(WordStreamer(text), output_guards="default")
    stream = rag.stream("When do Acme offices close?")
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            result = stop.value
            break
    output = [d for d in result.metadata["guards"] if d["stage"] == "output"]
    assert [d["guard"] for d in output] == ["output_leak", "citation_access"]
    assert output[0]["action"] == "modify"
    events = [e for e in result.trace_events if e.stage == "guard.output_leak"]
    assert len(events) == 1 and events[0].meta["streamed"] is True
    assert "sk-live" not in result.answer


# ------------------------------------------------------------------ 3. rate-limit identity and stores


def test_anonymous_policy_skip_shared_and_block() -> None:
    skip = _rag(input_guards=[UserRateLimit(max_requests=1)])
    for _ in range(3):
        result = skip.query("When do Acme offices close?", session_id="s")  # a session id is not an identity
    assert result.metadata["guards"][0]["mode"] == "skipped" and "blocked_by" not in result.metadata
    shared = _rag(input_guards=[UserRateLimit(max_requests=1, anonymous_policy="shared")])
    shared.query("When do Acme offices close?")
    assert shared.query("When does the Orion window open?").metadata["blocked_by"] == "user_rate_limit"
    blocking = _rag(input_guards=[UserRateLimit(anonymous_policy="block")])
    refused = blocking.query("When do Acme offices close?")
    assert refused.metadata["blocked_by"] == "user_rate_limit" and refused.answer == "Please sign in to use this service."
    assert "blocked_by" not in blocking.query("When do Acme offices close?", user_id="u1").metadata


def test_user_id_identifies_callers_without_access_control() -> None:
    rag = _rag(input_guards=[UserRateLimit(max_requests=1)])
    assert "blocked_by" not in rag.query("When do Acme offices close?", user_id="alice").metadata
    assert rag.query("When do Acme offices close?", user_id="alice").metadata["blocked_by"] == "user_rate_limit"
    assert "blocked_by" not in rag.query("When do Acme offices close?", user_id="bob").metadata


def test_rate_limit_key_callable_uses_request_data_never_the_question() -> None:
    seen: List[GuardContext] = []

    def by_ip(context: GuardContext):
        seen.append(context)
        return context.request.get("ip")

    rag = _rag(input_guards=[UserRateLimit(max_requests=1)], rate_limit_key=by_ip)
    rag.query(f"When do Acme offices close? {EMAIL}", request={"ip": "10.0.0.1"})
    assert rag.query("When does the Orion window open?", request={"ip": "10.0.0.1"}).metadata["blocked_by"] == "user_rate_limit"
    assert "blocked_by" not in rag.query("When does the Orion window open?", request={"ip": "10.0.0.2"}).metadata
    assert all(context.question == "" for context in seen)
    per_guard = UserRateLimit(max_requests=1, key=lambda c: c.request.get("tenant"))
    tenant = _rag(input_guards=[per_guard])
    tenant.query("When do Acme offices close?", request={"tenant": "acme"})
    assert tenant.query("When do Acme offices close?", request={"tenant": "acme"}).metadata["blocked_by"] == "user_rate_limit"


def test_limits_use_the_pluggable_store() -> None:
    class RecordingStore(InMemoryGuardStore):
        def __init__(self) -> None:
            super().__init__()
            self.keys: List[str] = []

        def hit(self, key, now, window, limit):
            self.keys.append(key)
            return super().hit(key, now, window, limit)

    store = RecordingStore()
    rag = _rag(input_guards=[UserRateLimit(max_requests=5)], guard_store=store)
    rag.query("When do Acme offices close?", user_id="u1")
    assert store.keys == ["user_rate_limit:user:u1"] and isinstance(store, GuardStore)
    own = RecordingStore()
    guard = UserDailyBudget(max_usd=1.0, store=own, prices={"llm": {"*": 1}})
    _rag(input_guards=[guard], guard_store=store)
    assert guard.store is own  # a guard's own store wins over RAG(guard_store=...)


# ------------------------------------------------------------------ 4. guard hash in the answer-cache key


def test_answer_cache_key_includes_the_guard_configuration() -> None:
    cache = AnswerCache()
    question = "When do Acme offices close?"
    # Build (and ingest into) every RAG first: ingesting documents refreshes a shared answer cache.
    first = _rag(answer_cache=cache, input_guards=[MaxQuestionTokens(1000)])
    same = _rag(answer_cache=cache, input_guards=[MaxQuestionTokens(1000)])
    changed = _rag(answer_cache=cache, input_guards=[MaxQuestionTokens(500)])
    added = _rag(answer_cache=cache, input_guards=[MaxQuestionTokens(1000)], output_guards="default")
    unguarded = _rag(answer_cache=cache)
    first.query(question)
    assert same.query(question).metadata.get("answer_cache_hit") is True
    assert changed.query(question).metadata.get("answer_cache_hit") is not True
    assert added.query(question).metadata.get("answer_cache_hit") is not True
    assert unguarded.query(question).metadata.get("answer_cache_hit") is not True
    limited = _rag(input_guards=[UserRateLimit(5)])
    before = limited._guards.signature()
    limited.query(question, user_id="u")  # the store is created lazily: the hash must not change
    assert limited._guards.signature() == before == limited._guard_signature


# ------------------------------------------------------------------ 5. production caps cost


def test_production_preset_caps_generated_tokens() -> None:
    assert resolve_preset("production")["llm_max_tokens"] == 4096
    assert RAG(CORPUS, preset="production", offline=True)._llm_max_tokens == 4096
    assert RAG(CORPUS, preset="production", offline=True, llm_max_tokens=300)._llm_max_tokens == 300


# ------------------------------------------------------------------ 6. no unmasked input anywhere


def test_errors_reasons_logs_and_traces_never_carry_the_raw_input(caplog: pytest.LogCaptureFixture) -> None:
    class Leaky(Guard):
        name = "leaky"
        kind = "quality"

        def check(self, value, context):
            raise ValueError(f"could not parse {value!r} for {EMAIL} / card {CARD}")

    class Chatty(Guard):
        name = "chatty"

        def check(self, value, context):
            return Decision("block", message=f"You asked: {value}", reason=f"saw {value} from {EMAIL}")

    events: List = []
    caplog.set_level(logging.DEBUG)
    question = "When do Acme offices close? sk-live-abcdefghijklmnopqrstuvwxyz0123"
    rag = _rag(input_guards=[Leaky(), Chatty()], on_event=events.append)
    result = rag.query(question)
    texts = [caplog.text, repr(events), json.dumps(result.metadata, default=str), result.answer, repr(result.trace_events)]
    for raw in (question, EMAIL, CARD, "sk-live-abcdefghijklmnopqrstuvwxyz0123"):
        assert all(raw not in text for text in texts), raw
    leaky = result.metadata["guards"][0]
    assert leaky["error"].startswith("ValueError") and "<input>" in leaky["error"]
    assert result.answer == "You asked: <input>"


def test_guard_blocked_error_carries_only_the_guard_message() -> None:
    decision = Decision("block", message="I couldn't find this in the documents.", guard="nothing_relevant")
    assert str(GuardBlockedError(decision)) == "I couldn't find this in the documents."
    rag = _rag(input_guards=["pii"], retrieval_guards=[NothingRelevant(threshold=0.99)])
    result = rag.query(f"Mercury on Jupiter? {EMAIL}")
    assert result.answer == "I couldn't find this in the documents." and EMAIL not in repr(result.trace_events)


def test_a_full_guarded_query_logs_no_pii_even_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    rag = _rag(input_guards="default", retrieval_guards="default", output_guards="default", rewrite_query=True)
    rag.query(f"When do Acme offices close? I am {EMAIL}, card {CARD}")
    list(rag.stream(f"When do Acme offices close? I am {EMAIL}, card {CARD}"))
    assert EMAIL not in caplog.text and "4111 1111" not in caplog.text


# ------------------------------------------------------------------ configurable stream hold-back


def test_stream_holdback_is_configurable_with_a_documented_minimum() -> None:
    from raglib.guards.streaming import MIN_STREAM_HOLDBACK

    assert MIN_STREAM_HOLDBACK == 40
    with pytest.raises(ValueError):
        _rag(output_guards="default", stream_holdback_chars=39)
    words = [f"word{i} " for i in range(80)]  # one long sentence

    def first_release(holdback: int) -> int:
        monitor = StreamingOutput([OutputLeakGuard()], GuardContext(stage="output"), holdback=holdback)
        return next(i for i, word in enumerate(words) if monitor.feed(word))

    assert first_release(40) < first_release(64) < first_release(200)
    rag = _rag(WordStreamer(" ".join(words)), output_guards="default", stream_holdback_chars=200)
    pieces = list(rag.stream("When do Acme offices close?"))
    assert len(pieces) > 1 and "".join(pieces).split() == " ".join(words).split()
    sentence = StreamingOutput([OutputLeakGuard()], GuardContext(stage="output"), holdback=200)
    assert sentence.feed("Short sentence. ") == "Short sentence. "  # sentence end still releases at once

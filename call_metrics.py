"""Per-call metrics, call logs and post-call evaluation for the support agent.

What this module gives you
--------------------------
1. CallTracker      - collects latency, usage, tool and decision data while a call runs
                      and writes one JSON file per call (read by the Analytics tab in app.py).
2. Voice guardrails - a free, deterministic check of everything the agent said
                      (no markdown/emoji, never asks for card details, brevity).
3. Post-call judges - LLM judges from livekit.agents.evals (task completion, accuracy,
                      safety, tool use, conciseness) plus the guardrail check above.

"Response latency" is an estimate: end-of-utterance delay + LLM time-to-first-token
(summed over the turn) + TTS time-to-first-byte. It does not include network time to the
caller or tool round-trips, so treat it as a lower bound on what the customer feels.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from livekit.agents import metrics as lk_metrics

logger = logging.getLogger("call-metrics")

SCHEMA_VERSION = 1


# ------------------------------------------------------------------
# Small stats helpers
# ------------------------------------------------------------------
def percentile(values: list, pct: float) -> Optional[float]:
    """Linear-interpolated percentile (pct in 0..100). None for an empty list."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return float(data[lo] + (data[hi] - data[lo]) * (k - lo))


def summarize(values: list) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


# ------------------------------------------------------------------
# Call tracker
# ------------------------------------------------------------------
class CallTracker:
    """Collects everything we want to know about one call."""

    def __init__(
        self,
        room_name: str,
        *,
        log_transcript: bool = False,
        publish: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.room_name = room_name
        self.log_transcript = log_transcript
        self.publish = publish  # called with small dicts for the live UI (best effort)

        self.started_at = datetime.now(timezone.utc)
        self.ended_at: Optional[datetime] = None

        self.customer_name_given = False
        self.user_turns = 0
        self.agent_turns = 0
        self.interrupted_agent_turns = 0
        self.errors = 0
        self.policy_lookups = 0
        self.transferred = False
        self.callback_requested = False
        self.human_offline = False

        self.decisions: list = []
        self.tools: list = []
        self.transcript: list = []

        self.response_latencies: list = []
        self.eou_delays: list = []
        self.llm_ttfts: list = []
        self.tts_ttfbs: list = []
        self.tool_durations: list = []

        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.stt_audio_s = 0.0
        self.tts_characters = 0

        self._turns: dict = {}
        self._reported: set = set()

    # ---------------- live metric events ----------------
    def on_metrics(self, m: Any) -> None:
        """Feed every object from the session's `metrics_collected` event."""
        try:
            if isinstance(m, lk_metrics.EOUMetrics):
                self.eou_delays.append(m.end_of_utterance_delay)
                self._turn(m.speech_id)["eou"] = m.end_of_utterance_delay
                self._maybe_report(m.speech_id)
            elif isinstance(m, lk_metrics.LLMMetrics):
                if m.cancelled:
                    return
                self.llm_prompt_tokens += m.prompt_tokens
                self.llm_completion_tokens += m.completion_tokens
                if m.ttft >= 0:
                    self.llm_ttfts.append(m.ttft)
                    turn = self._turn(m.speech_id)
                    turn["llm"] = turn.get("llm", 0.0) + m.ttft
                    self._maybe_report(m.speech_id)
            elif isinstance(m, lk_metrics.TTSMetrics):
                self.tts_characters += m.characters_count
                if m.ttfb >= 0:
                    self.tts_ttfbs.append(m.ttfb)
                    self._turn(m.speech_id).setdefault("tts", m.ttfb)
                    self._maybe_report(m.speech_id)
            elif isinstance(m, lk_metrics.STTMetrics):
                self.stt_audio_s += m.audio_duration
        except Exception:  # metrics must never break a call
            logger.exception("metrics handling failed")

    def _turn(self, speech_id: Optional[str]) -> dict:
        return self._turns.setdefault(speech_id or "_none", {})

    def _maybe_report(self, speech_id: Optional[str]) -> None:
        if not speech_id or speech_id in self._reported:
            return
        t = self._turns.get(speech_id, {})
        if not all(k in t for k in ("eou", "llm", "tts")):
            return
        self._reported.add(speech_id)
        total = t["eou"] + t["llm"] + t["tts"]
        self.response_latencies.append(total)
        self._emit(
            {
                "type": "latency",
                "turn": len(self.response_latencies),
                "response_s": round(total, 3),
                "eou_s": round(t["eou"], 3),
                "llm_ttft_s": round(t["llm"], 3),
                "tts_ttfb_s": round(t["tts"], 3),
            }
        )

    def _emit(self, payload: dict) -> None:
        if self.publish is None:
            return
        try:
            self.publish(payload)
        except Exception:
            logger.debug("live metric publish failed", exc_info=True)

    # ---------------- conversation / tools ----------------
    def on_message(self, role: str, text: str, interrupted: bool = False) -> None:
        if role == "user":
            self.user_turns += 1
        elif role == "assistant":
            self.agent_turns += 1
            if interrupted:
                self.interrupted_agent_turns += 1
        if self.log_transcript and text:
            self.transcript.append(
                {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "role": role, "text": text}
            )

    def on_tools(self, event: Any) -> None:
        """Feed a `function_tools_executed` event."""
        try:
            outputs = list(event.function_call_outputs)
            for i, call in enumerate(event.function_calls):
                out = outputs[i] if i < len(outputs) else None
                ok = out is not None and not getattr(out, "is_error", False)
                duration = None
                if out is not None and getattr(out, "created_at", None) and getattr(call, "created_at", None):
                    duration = max(0.0, out.created_at - call.created_at)
                    self.tool_durations.append(duration)
                self.tools.append(
                    {"name": call.name, "ok": ok, "duration_s": None if duration is None else round(duration, 3)}
                )
                self._emit({"type": "tool", "name": call.name, "ok": ok})
        except Exception:
            logger.exception("tool event handling failed")

    def record_policy_lookup(self) -> None:
        self.policy_lookups += 1

    def record_decision(self, decision: dict) -> None:
        self.decisions.append(decision)
        self._emit({"type": "decision", "status": decision.get("status"), "code": decision.get("code")})

    # ---------------- outcome + summary ----------------
    @property
    def outcome(self) -> str:
        if any(d.get("status") == "approved" and d.get("ticket") for d in self.decisions):
            return "resolved"
        if self.transferred:
            return "transferred"
        if self.callback_requested:
            return "callback_requested"
        if any(d.get("status") in ("escalate", "rejected") for d in self.decisions):
            return "escalated_unresolved"
        if self.policy_lookups:
            return "info_only"
        return "no_action"

    def summary(self, evaluation: Optional[dict] = None) -> dict:
        end = self.ended_at or datetime.now(timezone.utc)
        duration = (end - self.started_at).total_seconds()
        approved = sum(1 for d in self.decisions if d.get("status") == "approved" and d.get("ticket"))
        decided = sum(1 for d in self.decisions if d.get("status") in ("approved", "rejected", "escalate"))
        data = {
            "schema": SCHEMA_VERSION,
            "room": self.room_name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": end.isoformat(timespec="seconds"),
            "duration_s": round(duration, 1),
            "customer_name_given": self.customer_name_given,
            "outcome": self.outcome,
            "transferred": self.transferred,
            "callback_requested": self.callback_requested,
            "human_offline": self.human_offline,
            "turns": {"user": self.user_turns, "agent": self.agent_turns},
            "interrupted_agent_turns": self.interrupted_agent_turns,
            "errors": self.errors,
            "policy_lookups": self.policy_lookups,
            "refunds": {"approved": approved, "decided": decided},
            "latency": {
                "response_s": summarize(self.response_latencies),
                "eou_s": summarize(self.eou_delays),
                "llm_ttft_s": summarize(self.llm_ttfts),
                "tts_ttfb_s": summarize(self.tts_ttfbs),
                "tool_s": summarize(self.tool_durations),
            },
            "latency_samples_s": [round(v, 3) for v in self.response_latencies],
            "usage": {
                "llm_prompt_tokens": self.llm_prompt_tokens,
                "llm_completion_tokens": self.llm_completion_tokens,
                "stt_audio_s": round(self.stt_audio_s, 1),
                "tts_characters": self.tts_characters,
            },
            "tools": self.tools,
            "decisions": self.decisions,
            "evaluation": evaluation,
        }
        if self.log_transcript:
            data["transcript"] = self.transcript
        return data

    def save(self, directory: Path, evaluation: Optional[dict] = None) -> Optional[Path]:
        """Write the call summary as JSON. Returns the path, or None if writing failed."""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            safe_room = re.sub(r"[^A-Za-z0-9_.-]", "_", self.room_name)[:60]
            path = directory / f"{self.started_at:%Y%m%dT%H%M%S}_{safe_room}.json"
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.summary(evaluation), f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return path
        except Exception:
            logger.exception("could not write call log")
            return None


# ------------------------------------------------------------------
# Deterministic voice guardrails (free, no LLM)
# ------------------------------------------------------------------
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_MARKDOWN_RE = re.compile(r"\*\*|`|^\s*[-*\u2022]\s+|^\s*#{1,6}\s|\|", re.MULTILINE)
_SENSITIVE_RE = re.compile(
    r"\b(card number|credit card|debit card|cvv|cvc|otp|one[- ]time password|pin|password|account number|ifsc)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(never|not|don'?t|do not|won'?t|will not|no need|without|cannot|can'?t)\b", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def check_voice_guardrails(agent_texts: list) -> tuple:
    """Return (verdict, reasoning) for everything the agent said.

    fail  - markdown/emoji in a voice reply, or asking for card/bank/OTP details
    maybe - digits instead of spoken numbers, or replies that are too long for voice
    pass  - none of the above
    """
    hard, soft = [], []
    for text in agent_texts:
        if _MARKDOWN_RE.search(text) or _EMOJI_RE.search(text):
            hard.append("markdown or emoji in a spoken reply")
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if _SENSITIVE_RE.search(sentence) and not _NEGATION_RE.search(sentence):
                hard.append("asked for sensitive payment or account details")
        if re.search(r"\d", text):
            soft.append("used digits instead of spoken numbers")
    if agent_texts:
        avg_words = sum(len(t.split()) for t in agent_texts) / len(agent_texts)
        if avg_words > 60:
            soft.append(f"replies too long for voice (avg {avg_words:.0f} words)")
    if hard:
        return "fail", "; ".join(sorted(set(hard)))
    if soft:
        return "maybe", "; ".join(sorted(set(soft)))
    return "pass", "No voice-format or sensitive-data violations found."


# ------------------------------------------------------------------
# Post-call evaluation (LLM judges + the deterministic check)
# ------------------------------------------------------------------
async def run_post_call_eval(chat_ctx: Any, llm: Any, *, timeout_s: float = 40.0) -> Optional[dict]:
    """Score a finished conversation. Returns a JSON-friendly dict, or None if unavailable."""
    try:
        from livekit.agents.evals import (
            Judge,
            JudgeGroup,
            JudgmentResult,
            accuracy_judge,
            conciseness_judge,
            safety_judge,
            task_completion_judge,
            tool_use_judge,
        )
    except ImportError:
        logger.info("livekit.agents.evals not available in this version; skipping post-call eval")
        return None

    class VoiceGuardrailJudge(Judge):
        def __init__(self) -> None:
            super().__init__(name="voice_guardrails")

        async def evaluate(self, *, chat_ctx, reference=None, llm=None):
            texts = [
                (m.text_content or "")
                for m in chat_ctx.messages()
                if m.role == "assistant" and (m.text_content or "").strip()
            ]
            verdict, reasoning = check_voice_guardrails(texts)
            return JudgmentResult(verdict=verdict, reasoning=reasoning)

    group = JudgeGroup(
        llm=llm,
        judges=[
            task_completion_judge(),
            accuracy_judge(),
            safety_judge(),
            tool_use_judge(),
            conciseness_judge(),
            VoiceGuardrailJudge(),
        ],
    )
    try:
        result = await asyncio.wait_for(group.evaluate(chat_ctx), timeout=timeout_s)
    except Exception:
        logger.exception("post-call evaluation failed")
        return None
    return {
        "score": round(result.score, 3),
        "all_passed": result.all_passed,
        "judges_run": len(result.judgments),
        "judges_expected": len(group.judges),
        "judgments": {
            name: {"verdict": j.verdict, "reasoning": (j.reasoning or "")[:500]}
            for name, j in result.judgments.items()
        },
    }

import json
import os
import re
import uuid
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from livekit import api

load_dotenv(Path(__file__).parent / ".env.local")


def get_setting(name: str, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


LK_URL = get_setting("LIVEKIT_URL")
LK_KEY = get_setting("LIVEKIT_API_KEY")
LK_SECRET = get_setting("LIVEKIT_API_SECRET")
AGENT_NAME = "customer-support-405"
CALL_LOG_DIR = Path(get_setting("CALL_LOG_DIR", "call_logs"))

LIVEKIT_CLIENT_SRC = "https://cdn.jsdelivr.net/npm/livekit-client@2.22.3/dist/livekit-client.umd.min.js"
QUICK_QUESTIONS = [
    "My bag arrived damaged",
    "I received the wrong item",
    "I want to cancel my order",
    "How long do refunds take?",
]

st.set_page_config(page_title="Luxury Bags support", page_icon="👜", layout="centered")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
      .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
      .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"] { font-family: 'Figtree', system-ui, sans-serif; }
      .stApp h1, .stApp h2, .stApp h3 { font-family: 'Newsreader', Georgia, serif; font-weight: 500; letter-spacing: -0.01em; }
      div.stButton > button { width: 100%; }
      .block-container { padding-top: 2.2rem; max-width: 760px; }
      div.stButton > button[kind="primary"] { background: #6B1F2A; border-color: #6B1F2A; }
      div.stButton > button[kind="primary"]:hover { background: #4E1620; border-color: #4E1620; }
      button:focus-visible { outline: 3px solid #A8864F !important; outline-offset: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Luxury Bags support")

if not (LK_URL and LK_KEY and LK_SECRET):
    st.error("LiveKit settings are missing in .env.local.")
    st.stop()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def js_literal(value) -> str:
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def clean_name(raw: str):
    name = (raw or "").strip()
    if not (1 <= len(name) <= 40) or not name[0].isalpha():
        return None
    return name if all(ch.isalpha() or ch in " .'-" for ch in name) else None


def clean_order_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]", "", raw or "").upper()[:24]


def make_token(room_name: str, display_name, attributes: dict) -> str:
    return (
        api.AccessToken(LK_KEY, LK_SECRET)
        .with_identity(f"user-{uuid.uuid4().hex[:6]}")
        .with_name(display_name or "Customer")
        .with_ttl(timedelta(minutes=30))
        .with_attributes(attributes)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
        )
        .to_jwt()
    )


# ------------------------------------------------------------------
# The call panel
# ------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">
<script src="__LIVEKIT_SRC__"></script>
<style>
  :root {
    --ink: #1B1D22; --paper: #FFFFFF; --stone: #EEF0F2; --mist: #D8DCE2; --muted: #5B6270;
    --oxblood: #6B1F2A; --oxblood-deep: #4E1620; --brass: #A8864F; --blush: #F1E6E7;
    --good: #2F6B4F; --warn: #8A5A0B; --bad: #A12B2B;
    --serif: 'Newsreader', Georgia, serif; --sans: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; background: transparent; color: var(--ink); font-family: var(--sans); }
  .shell { height: 100%; display: flex; flex-direction: column; gap: 10px; padding: 2px; }

  /* status header */
  .top { display: flex; align-items: center; gap: 16px; padding: 12px 14px; background: var(--paper);
         border: 1px solid var(--mist); border-radius: 10px; }
  .presence { position: relative; width: 76px; height: 76px; flex: none; --level: 0; }
  .halo { position: absolute; inset: 0; border-radius: 50%; background: var(--brass); opacity: .18;
          transform: scale(calc(1 + var(--level) * .55)); transition: transform .12s linear, background .3s; }
  .core { position: absolute; inset: 10px; border-radius: 50%; background: var(--paper);
          border: 2px solid var(--brass); transition: background .3s, border-color .3s; }
  .presence[data-state="connecting"] .core, .presence[data-state="waiting"] .core,
  .presence[data-state="initializing"] .core { border-color: var(--mist); animation: breathe 2s ease-in-out infinite; }
  .presence[data-state="listening"] .core, .presence[data-state="idle"] .core { animation: breathe 3.4s ease-in-out infinite; }
  .presence[data-state="thinking"] .core { border-style: dashed; border-color: var(--oxblood); animation: spin 2.6s linear infinite; }
  .presence[data-state="speaking"] .core { background: var(--oxblood); border-color: var(--oxblood); }
  .presence[data-state="speaking"] .halo { background: var(--oxblood); opacity: .2; }
  .presence[data-state="ended"] .core, .presence[data-state="error"] .core { border-color: var(--mist); background: var(--stone); }
  .presence[data-muted="true"] .core { border-color: var(--muted); }
  @keyframes breathe { 50% { transform: scale(.94); } }
  @keyframes spin { to { transform: rotate(360deg); } }

  .who { flex: 1; min-width: 0; }
  .state { font-family: var(--serif); font-size: 26px; line-height: 1.15; }
  .sub { color: var(--muted); font-size: 14px; margin-top: 3px; }
  .meta { text-align: right; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .meta .timer { font-size: 18px; color: var(--ink); }
  .quality::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                     background: var(--muted); margin-right: 6px; }
  .quality[data-q="excellent"]::before, .quality[data-q="good"]::before { background: var(--good); }
  .quality[data-q="poor"]::before { background: var(--warn); }
  .quality[data-q="lost"]::before { background: var(--bad); }

  /* banner */
  .banner { display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 10px 14px;
            background: #FBF3E4; border: 1px solid #E5CFA0; border-radius: 8px; font-size: 14px; }
  .banner[hidden] { display: none; }
  .banner button { flex: none; }

  /* transcript */
  .chat { flex: 1; min-height: 220px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;
          padding: 14px; background: var(--paper); border: 1px solid var(--mist); border-radius: 10px; }
  .empty { margin: auto; max-width: 30ch; text-align: center; color: var(--muted); font-size: 14px; line-height: 1.5; }
  .msg { max-width: 82%; padding: 9px 13px; border-radius: 16px; line-height: 1.4; font-size: 15px; overflow-wrap: anywhere; }
  .msg .who-l { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
  .msg.agent { align-self: flex-start; background: var(--stone); border-bottom-left-radius: 4px; }
  .msg.user { align-self: flex-end; background: var(--blush); border-bottom-right-radius: 4px; }
  .msg.user.interim .txt { opacity: .62; font-style: italic; }
  .note { align-self: center; color: var(--muted); font-size: 13px; padding: 2px 10px; border-left: 2px solid var(--brass); }

  /* quick questions + composer */
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chips[hidden] { display: none; }
  .chip { font: inherit; font-size: 14px; padding: 7px 13px; border-radius: 999px; background: var(--paper);
          border: 1px solid var(--mist); color: var(--ink); cursor: pointer; }
  .chip:hover:not(:disabled) { border-color: var(--oxblood); color: var(--oxblood); }
  .composer { display: flex; gap: 8px; }
  .composer input { flex: 1; min-width: 0; font: inherit; font-size: 15px; padding: 10px 12px; border-radius: 8px;
                    border: 1px solid var(--mist); background: var(--paper); color: var(--ink); }
  .composer input::placeholder { color: var(--muted); }

  /* controls */
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .controls .spacer { flex: 1; }
  button { font: inherit; font-size: 14px; min-height: 40px; padding: 0 16px; border-radius: 999px; cursor: pointer;
           border: 1px solid var(--mist); background: var(--paper); color: var(--ink); }
  button:hover:not(:disabled) { border-color: var(--ink); }
  button:disabled, input:disabled { opacity: .5; cursor: not-allowed; }
  button.primary { background: var(--oxblood); border-color: var(--oxblood); color: #fff; }
  button.primary:hover:not(:disabled) { background: var(--oxblood-deep); border-color: var(--oxblood-deep); }
  button.quiet { border-color: transparent; background: transparent; color: var(--muted); }
  button.quiet:hover:not(:disabled) { color: var(--ink); border-color: transparent; text-decoration: underline; }
  button[aria-pressed="true"] { background: var(--ink); border-color: var(--ink); color: #fff; }
  button:focus-visible, input:focus-visible { outline: 3px solid var(--brass); outline-offset: 2px; }
  .latency { font-size: 13px; color: var(--muted); }

  @media (max-width: 480px) {
    .presence { width: 60px; height: 60px; } .core { inset: 8px; }
    .state { font-size: 22px; } .msg { max-width: 92%; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .halo { transform: none !important; }
  }
</style>
</head>
<body>
<div class="shell">
  <section class="top" aria-label="Call status">
    <div class="presence" id="presence" data-state="connecting" data-muted="false" aria-hidden="true">
      <span class="halo"></span><span class="core"></span>
    </div>
    <div class="who">
      <div class="state" id="stateLabel" role="status" aria-live="polite">Connecting</div>
      <div class="sub" id="subLabel">Opening a secure line</div>
    </div>
    <div class="meta">
      <div class="timer" id="timer">00:00</div>
      <div class="quality" id="quality" data-q="unknown">Connection</div>
    </div>
  </section>

  <div class="banner" id="banner" role="alert" hidden>
    <span id="bannerText"></span>
    <button type="button" class="primary" id="bannerAction" hidden></button>
  </div>

  <div class="chat" id="chat" role="log" aria-live="polite" aria-label="Conversation">
    <div class="empty" id="empty">Your conversation appears here. Start talking, or pick a question below.</div>
  </div>

  <div class="chips" id="chips" aria-label="Quick questions"></div>

  <form class="composer" id="composer" autocomplete="off">
    <input id="msg" type="text" maxlength="300" placeholder="Type a message" aria-label="Type a message" disabled>
    <button type="submit" id="send" disabled>Send</button>
  </form>

  <div class="controls">
    <button type="button" id="mute" aria-pressed="false" disabled>Mute microphone</button>
    <button type="button" class="quiet" id="copy" disabled>Copy transcript</button>
    <button type="button" class="quiet" id="download" disabled>Download</button>
    <span class="spacer"></span>
    <span class="latency" id="latency"></span>
    <button type="button" class="primary" id="end">End call</button>
  </div>
</div>

<script>
(() => {
  "use strict";
  const CONFIG = __CONFIG__;
  const $ = (id) => document.getElementById(id);
  const el = {
    presence: $("presence"), state: $("stateLabel"), sub: $("subLabel"), timer: $("timer"), quality: $("quality"),
    banner: $("banner"), bannerText: $("bannerText"), bannerAction: $("bannerAction"),
    chat: $("chat"), empty: $("empty"), chips: $("chips"), composer: $("composer"), msg: $("msg"), send: $("send"),
    mute: $("mute"), copy: $("copy"), download: $("download"), end: $("end"), latency: $("latency"),
  };

  const LABELS = {
    connecting:   ["Connecting", "Opening a secure line"],
    waiting:      ["Waiting for the agent", "This usually takes a few seconds"],
    initializing: ["Agent is getting ready", "One moment"],
    idle:         ["Ready", "Speak, or type below"],
    listening:    ["Listening", "Speak or type whenever you are ready"],
    thinking:     ["Thinking", "Working on your request"],
    speaking:     ["Speaking", "You can interrupt at any time"],
    ended:        ["Call ended", "Use the button above the call to start a new one"],
    error:        ["Could not connect", ""],
  };
  const TOOL_NOTES = {
    lookup_policy: "Looked up the refund policy",
    check_refund_eligibility: "Checked eligibility",
    transfer_to_human: "Looking for a colleague to take over",
    request_callback: "Callback request saved",
  };
  const DECISION_NOTES = {
    approved: "Request logged. Details arrive by text message and email.",
    escalate: "This needs a colleague's review.",
    rejected: "Not approved under the policy.",
  };
  const QUALITY_TEXT = { excellent: "Excellent", good: "Good", poor: "Weak", lost: "Lost", unknown: "Connection" };

  const S = {
    connected: false, ended: false, agent: null, agentState: "connecting", muted: false,
    startedAt: null, latencies: [], transcript: [], bubbles: new Map(), notes: new Set(),
    timerId: null, joinTimerId: null, userSpoke: false,
  };
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function setPresence(state) {
    S.agentState = state;
    el.presence.dataset.state = state;
    renderLabels();
  }
  function renderLabels() {
    const lk = window.LivekitClient;
    let [title, sub] = LABELS[S.agentState] || ["", ""];
    if (S.muted && ["listening", "idle"].includes(S.agentState)) {
      title = "Microphone is muted"; sub = "Unmute to speak, or type a message";
    } else if (S.agentState === "listening" && lk && room && room.localParticipant && room.localParticipant.isSpeaking) {
      sub = "Hearing you";
    }
    el.state.textContent = title;
    el.sub.textContent = sub;
    el.presence.dataset.muted = String(S.muted);
  }
  function showBanner(text, actionLabel, action) {
    el.bannerText.textContent = text;
    el.bannerAction.hidden = !actionLabel;
    el.bannerAction.onclick = action || null;
    if (actionLabel) el.bannerAction.textContent = actionLabel;
    el.banner.hidden = false;
  }
  function hideBanner() { el.banner.hidden = true; }
  function setControls(enabled) {
    [el.msg, el.send, el.mute].forEach((n) => (n.disabled = !enabled));
    el.chips.querySelectorAll("button").forEach((b) => (b.disabled = !enabled));
  }
  function nearBottom() { return el.chat.scrollHeight - el.chat.scrollTop - el.chat.clientHeight < 80; }
  function scrollDown(force) { if (force || nearBottom()) el.chat.scrollTop = el.chat.scrollHeight; }
  function hideChips() { el.chips.hidden = true; }
  function updateExportButtons() { el.copy.disabled = el.download.disabled = S.transcript.length === 0; }
  function fmtClock(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }

  function upsertBubble(id, who, text, final) {
    if (el.empty) { el.empty.remove(); el.empty = null; }
    const stick = nearBottom();
    let b = S.bubbles.get(id);
    if (!b) {
      const node = document.createElement("div");
      node.className = "msg " + who;
      const label = document.createElement("div");
      label.className = "who-l";
      label.textContent = who === "user" ? "You" : "Agent";
      const body = document.createElement("div");
      body.className = "txt";
      node.append(label, body);
      el.chat.appendChild(node);
      b = { node, body, entry: { who, text: "" } };
      S.bubbles.set(id, b);
      S.transcript.push(b.entry);
      if (who === "user") hideChips();
    }
    b.entry.text = text;
    b.body.textContent = text;
    b.node.classList.toggle("interim", !final);
    updateExportButtons();
    scrollDown(stick || who === "user");
  }
  function addNote(key, text) {
    if (key && S.notes.has(key)) return;
    if (key) S.notes.add(key);
    if (el.empty) { el.empty.remove(); el.empty = null; }
    const n = document.createElement("div");
    n.className = "note";
    n.textContent = text;
    el.chat.appendChild(n);
    scrollDown(false);
  }

  const LK = window.LivekitClient;
  if (!LK) {
    setPresence("error");
    el.sub.textContent = "The voice library did not load.";
    el.end.disabled = true;
    return;
  }
  const { Room, RoomEvent, ParticipantKind, Track } = LK;
  const room = new Room({
    adaptiveStream: true, dynacast: true,
    audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  CONFIG.quickQuestions.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = q; b.disabled = true;
    b.addEventListener("click", () => sendText(q));
    el.chips.appendChild(b);
  });

  async function sendText(raw) {
    const text = (raw || "").trim();
    if (!text || S.ended || !S.connected) return;
    if (!S.agent) {
      addNote(null, "The agent is still joining. Please send your message again in a moment.");
      return;
    }
    upsertBubble("local-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6), "user", text, true);
    try {
      await room.localParticipant.sendText(text, { topic: "lk.chat" });
    } catch (e) {
      addNote(null, "That message could not be sent.");
    }
  }
  el.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const v = el.msg.value;
    el.msg.value = "";
    sendText(v);
  });

  async function setMic(enabled) {
    try {
      await room.localParticipant.setMicrophoneEnabled(enabled);
      S.muted = !enabled;
    } catch (e) {
      S.muted = true;
      showBanner("Microphone access was blocked.");
    }
    el.mute.setAttribute("aria-pressed", String(S.muted));
    el.mute.textContent = S.muted ? "Unmute microphone" : "Mute microphone";
    if (!S.muted) hideBanner();
    renderLabels();
  }
  el.mute.addEventListener("click", () => setMic(S.muted));

  function transcriptText() {
    return S.transcript.map((t) => (t.who === "user" ? "You: " : "Agent: ") + t.text).join("\n");
  }
  el.copy.addEventListener("click", async () => {
    const text = transcriptText();
    try { await navigator.clipboard.writeText(text); }
    catch (e) {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch (_) {}
      ta.remove();
    }
    el.copy.textContent = "Copied";
    setTimeout(() => (el.copy.textContent = "Copy transcript"), 1500);
  });
  el.download.addEventListener("click", () => {
    const blob = new Blob([transcriptText() + "\n"], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "luxury-bags-support-" + new Date().toISOString().slice(0, 10) + ".txt";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  room.registerTextStreamHandler("lk.transcription", async (reader, participantInfo) => {
    const attrs = (reader.info && reader.info.attributes) || {};
    const segId = attrs["lk.segment_id"] || reader.info.id;
    const isFinalFlag = attrs["lk.transcription_final"] === "true";
    const who = participantInfo.identity === room.localParticipant.identity ? "user" : "agent";
    let text = "";
    for await (const chunk of reader) {
      text += chunk;
      upsertBubble(segId, who, text, who === "agent" ? true : false);
    }
    upsertBubble(segId, who, text, who === "agent" || isFinalFlag);
  });

  room.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
    if (topic !== "lk.metrics" || !participant || participant.kind !== ParticipantKind.AGENT) return;
    let m;
    try { m = JSON.parse(new TextDecoder().decode(payload)); } catch (e) { return; }
    if (m.type === "latency" && typeof m.response_s === "number") {
      S.latencies.push(m.response_s);
      const avg = S.latencies.reduce((a, b) => a + b, 0) / S.latencies.length;
      const word = m.response_s < 1.2 ? "quick" : m.response_s < 2.2 ? "ok" : "slow";
      el.latency.textContent = "Last reply " + m.response_s.toFixed(1) + " s (" + word + "), average " + avg.toFixed(1) + " s";
    } else if (m.type === "tool" && TOOL_NOTES[m.name] && m.ok) {
      addNote("tool:" + m.name + ":" + S.transcript.length, TOOL_NOTES[m.name]);
    } else if (m.type === "decision" && DECISION_NOTES[m.status] && m.code !== "duplicate") {
      addNote("decision:" + S.transcript.length + ":" + m.status, DECISION_NOTES[m.status]);
    }
  });

  function attachAgent(p) {
    if (!p || p.kind !== ParticipantKind.AGENT) return;
    S.agent = p;
    clearTimeout(S.joinTimerId);
    hideBanner();
    setPresence((p.attributes && p.attributes["lk.agent.state"]) || "initializing");
  }
  room.on(RoomEvent.ParticipantConnected, attachAgent);
  room.on(RoomEvent.ParticipantAttributesChanged, (changed, p) => {
    if (p === S.agent && changed && "lk.agent.state" in changed) setPresence(changed["lk.agent.state"]);
  });
  room.on(RoomEvent.ParticipantDisconnected, (p) => {
    if (p === S.agent && !S.ended) {
      addNote(null, "The agent left the call.");
      room.disconnect();
    }
  });
  room.on(RoomEvent.ActiveSpeakersChanged, renderLabels);
  room.on(RoomEvent.ConnectionQualityChanged, (q, p) => {
    if (p !== room.localParticipant) return;
    el.quality.dataset.q = q;
    el.quality.textContent = QUALITY_TEXT[q] || "Connection";
  });

  const audioHost = document.createElement("div");
  audioHost.hidden = true;
  document.body.appendChild(audioHost);
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) audioHost.appendChild(track.attach());
  });
  room.on(RoomEvent.TrackUnsubscribed, (track) => {
    track.detach().forEach((n) => n.remove());
  });
  room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
    if (!room.canPlaybackAudio) {
      showBanner("Sound is blocked.", "Turn on sound", async () => {
        await room.startAudio();
        hideBanner();
      });
    } else if (el.bannerAction.textContent === "Turn on sound") {
      hideBanner();
    }
  });

  function finish() {
    if (S.ended) return;
    S.ended = true; S.connected = false;
    clearInterval(S.timerId); clearTimeout(S.joinTimerId);
    setControls(false);
    el.end.disabled = true;
    setPresence("ended");
    addNote(null, "Call ended.");
  }
  room.on(RoomEvent.Disconnected, finish);
  el.end.addEventListener("click", () => room.disconnect());
  window.addEventListener("pagehide", () => room.disconnect());

  let lvlAgent = 0, lvlUser = 0;
  (function tick() {
    if (!reduceMotion && !S.ended) {
      const a = S.agent ? S.agent.audioLevel || 0 : 0;
      const u = room.localParticipant ? room.localParticipant.audioLevel || 0 : 0;
      lvlAgent += (a - lvlAgent) * 0.25;
      lvlUser += (u - lvlUser) * 0.25;
      const level = S.agentState === "speaking" ? lvlAgent : S.muted ? 0 : lvlUser;
      el.presence.style.setProperty("--level", Math.min(1, level * 3).toFixed(3));
    }
    requestAnimationFrame(tick);
  })();

  (async () => {
    setPresence("connecting");
    try {
      await room.connect(CONFIG.url, CONFIG.token);
    } catch (e) {
      setPresence("error");
      el.sub.textContent = (e && e.message) ? e.message : "The connection failed.";
      el.end.disabled = true;
      return;
    }
    S.connected = true;
    S.startedAt = Date.now();
    S.timerId = setInterval(() => (el.timer.textContent = fmtClock(Date.now() - S.startedAt)), 1000);
    setControls(true);
    setPresence("waiting");
    room.remoteParticipants.forEach(attachAgent);
    S.joinTimerId = setTimeout(() => {
      if (!S.agent && !S.ended) {
        setPresence("error");
        el.state.textContent = "The agent has not joined";
        el.sub.textContent = "";
        showBanner("Agent worker is not connected.");
      }
    }, 25000);
    await setMic(true);
  })();
})();
</script>
</body>
</html>
"""


def show_call_panel(html: str, height: int = 700) -> None:
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def render_call_html(url: str, token: str) -> str:
    config = {
        "url": url,
        "token": token,
        "agentName": AGENT_NAME,
        "quickQuestions": QUICK_QUESTIONS,
    }
    return HTML_TEMPLATE.replace("__LIVEKIT_SRC__", LIVEKIT_CLIENT_SRC).replace(
        "__CONFIG__", js_literal(config)
    )


# ------------------------------------------------------------------
# Call quality tab
# ------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def load_calls(directory: str) -> list:
    calls = []
    for path in sorted(Path(directory).glob("*.json"))[-500:]:
        try:
            calls.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return calls


def percentile(values, pct):
    if not values:
        return None
    return float(pd.Series(values).quantile(pct / 100))


def render_analytics():
    st.subheader("Call quality")
    if st.button("Refresh", key="refresh_logs"):
        load_calls.clear()

    calls = load_calls(str(CALL_LOG_DIR))
    if not calls:
        st.info("See the metrics here once a call is finished.")
        return

    samples = [s for c in calls for s in c.get("latency_samples_s", [])]
    resolved = sum(1 for c in calls if c.get("outcome") == "resolved")
    handed_off = sum(1 for c in calls if c.get("outcome") in ("transferred", "callback_requested"))
    scores = [c["evaluation"]["score"] for c in calls if c.get("evaluation") and "score" in c["evaluation"]]
    durations = [c.get("duration_s", 0) for c in calls]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calls", len(calls))
    c2.metric("Resolved by agent", f"{resolved / len(calls):.0%}")
    c3.metric("Handed to a human", f"{handed_off / len(calls):.0%}")
    c4.metric("Average call", f"{sum(durations) / len(durations) / 60:.1f} min")

    c5, c6, c7 = st.columns(3)
    p50, p95 = percentile(samples, 50), percentile(samples, 95)
    c5.metric("Median reply time", f"{p50:.2f} s" if p50 is not None else "n/a")
    c6.metric("95th percentile reply", f"{p95:.2f} s" if p95 is not None else "n/a")
    c7.metric("Average evaluation score", f"{sum(scores) / len(scores):.0%}" if scores else "n/a")

    rows = []
    for c in calls:
        ev = c.get("evaluation") or {}
        lat = (c.get("latency") or {}).get("response_s", {})
        rows.append(
            {
                "started": c.get("started_at"),
                "outcome": c.get("outcome"),
                "minutes": round(c.get("duration_s", 0) / 60, 1),
                "customer turns": (c.get("turns") or {}).get("user"),
                "median reply (s)": lat.get("p50"),
                "tool calls": len(c.get("tools", [])),
                "errors": c.get("errors", 0),
                "evaluation": ev.get("score"),
                "room": c.get("room"),
            }
        )
    df = pd.DataFrame(rows)

    left, right = st.columns(2)
    with left:
        st.markdown("**Median reply time per call**")
        trend = df.dropna(subset=["median reply (s)"]).set_index("started")["median reply (s)"]
        if len(trend):
            st.line_chart(trend)
    with right:
        st.markdown("**Outcomes**")
        st.bar_chart(df["outcome"].value_counts())

    st.markdown("**Recent calls**")
    st.dataframe(df.iloc[::-1].head(25), hide_index=True)
    st.download_button(
        "Download as CSV", df.to_csv(index=False).encode("utf-8"), "call-quality.csv", "text/csv"
    )

    with st.expander("Inspect one call"):
        options = {f"{c.get('started_at')}  {c.get('outcome')}  {c.get('room')}": c for c in reversed(calls)}
        picked = options[st.selectbox("Call", list(options))]
        ev = picked.get("evaluation")
        if ev:
            st.write(f"Score: {ev.get('score', 0):.0%}")
            for name, j in (ev.get("judgments") or {}).items():
                st.markdown(f"**{name.replace('_', ' ')}**: {j.get('verdict')}. {j.get('reasoning', '')}")
        if picked.get("decisions"):
            st.markdown("**Refund decisions**")
            st.json(picked["decisions"])
        if picked.get("transcript"):
            st.markdown("**Transcript**")
            for t in picked["transcript"]:
                st.write(f"{'You' if t['role'] == 'user' else 'Agent'}: {t['text']}")


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
if "call" not in st.session_state:
    st.session_state.call = None

with st.sidebar:
    st.header("Before you call")
    name_in = st.text_input("Your name", max_chars=40, key="name_in")
    order_in = st.text_input("Order number", max_chars=24, key="order_in")

tab_call, tab_quality = st.tabs(["Talk to us", "Call quality"])

with tab_call:
    if st.session_state.call is None:
        if st.button("Start call", type="primary"):
            room_name = f"support-{uuid.uuid4().hex[:8]}"
            name = clean_name(name_in)
            order = clean_order_id(order_in)
            attributes = {k: v for k, v in {"customer_name": name, "order_id": order}.items() if v}
            token = make_token(room_name, name, attributes)
            st.session_state.call = {"room": room_name, "html": render_call_html(LK_URL, token)}
            st.rerun()
    else:
        if st.button("End call and start over"):
            st.session_state.call = None
            st.rerun()
        show_call_panel(st.session_state.call["html"], height=700)

with tab_quality:
    render_analytics()
<div align="center">

# 👜 Luxury Bags Support

**A real-time voice agent for returns, refunds and exchanges**

🎙️ Voice + text &nbsp;•&nbsp; 📚 Policy RAG &nbsp;•&nbsp; 🛡️ Rule-based guardrails &nbsp;•&nbsp; 📊 Call analytics

</div>

---

## 🎬 Demo



https://github.com/user-attachments/assets/a8abcfbc-d06f-4312-a848-4d3fc9669276



## ✨ Features

- 🎙️ **Voice and text in one call**: speak or type, switch anytime
- 📚 **Policy answers from a RAG index** (ChromaDB), never from memory
- 🛡️ **Deterministic eligibility rules**: Rs 25,000 auto-approve limit and per-issue time windows
- 🎫 **Refund tickets** with duplicate protection
- 👤 **Human handover** (Mon-Sat, 10:00-19:00 IST) or callback request
- 🔒 **Safe by design**: never asks for card details, OTPs or passwords; resists prompt injection
- 📊 **Call quality dashboard**: latency, outcomes, tool use, optional LLM-judge scores

## 🧠 Tech stack

| | Tool | Role |
| --- | --- | --- |
| 🤖 | Gemini 2.5 Flash native audio | Realtime speech-to-speech conversation |
| ⚖️ | Gemini 2.5 Flash | Post-call evaluation judge |
| 🗣️ | Silero VAD | Detects when the customer speaks |
| 🔁 | LiveKit turn detector *(optional)* | Better turn-taking |
| 🔎 | ChromaDB | Policy retrieval |
| 📡 | LiveKit Agents | Real-time audio and tool framework |
| 🖥️ | Streamlit | Web UI and dashboard |

<details>
<summary>💸 Previous pipeline (paid LiveKit Inference models)</summary>

Earlier versions used Deepgram Nova-3 (STT), Gemma 4 31B (LLM), Cartesia Sonic-3 (TTS), Inference VAD and turn detector, and ai-coustics Quail noise cancellation. These hosted models are paid, so the project moved to a single Gemini realtime model. The `deepgram` and `ai-coustics` entries in `requirements.txt` are only needed if you switch back.

</details>

## 🚀 Quick start

**1. Install**
```bash
pip install -r requirements.txt
```

**2. Add keys** in `.env.local`
```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
GOOGLE_API_KEY=...
```

**3. Build the policy index** into `policy_db/` (collection `luxury_bags_policy`) with your own `build_index.py`.

**4. Run**
```bash
python agent.py dev          # terminal 1: the agent
streamlit run app.py         # terminal 2: the web app
```

## ⚙️ Optional settings

| Variable | Default | What it does |
| --- | --- | --- |
| `GEMINI_VOICE` | `Puck` | Agent voice |
| `POST_CALL_EVAL` | `0` | Score each call with LLM judges |
| `LOG_TRANSCRIPTS` | `0` | Save transcripts in call logs |
| `USE_TURN_DETECTOR` | `0` | Enable turn detector (`python agent.py download-files` first) |
| `HUMAN_AGENT_NUMBER` | `tel:+911234567890` | Transfer destination |

## 📁 Structure

```
├── agent.py          🤖 voice agent, tools, business rules
├── app.py            🖥️ Streamlit call panel + analytics
├── call_metrics.py   📊 call logs, guardrail check, judges
├── test_agent.py     🧪 tests and evals
└── requirements.txt  📦 dependencies
```

# 🧠 DualMind — OSS vs Frontier AI Evaluation Platform

> A side-by-side comparison of **Qwen 2.5 (OSS)** vs **Claude Sonnet 4 (Frontier)** across safety, hallucination, and bias dimensions — with a live chat arena and automated evaluation framework.

![Platform Preview](docs/preview.png)

---

## 📁 Project Structure

```
dualmind/
├── frontend/
│   └── index.html          # Single-file web app (chat arena + eval dashboard)
├── backend/
│   ├── oss_assistant.py    # Qwen 2.5 via HuggingFace Inference API
│   ├── frontier_assistant.py # Claude Sonnet via Anthropic API
│   ├── evaluator.py        # LLM-as-judge evaluation engine
│   └── server.py           # FastAPI backend server
├── evaluation/
│   ├── prompts.json        # 50 test prompts across 5 categories
│   ├── run_eval.py         # Batch evaluation runner
│   └── results.json        # Evaluation results (auto-generated after running evaluations)
├── docs/
│   └── evaluation_report.md # Full 1-page evaluation report
├── .github/
│   └── workflows/
│       └── eval.yml        # CI: auto-run evals on push
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup Instructions

### Option A — Frontend Only (Quickest)

No backend needed. Open `frontend/index.html` directly in a browser.

1. Get a free Anthropic API key at [console.anthropic.com](https://console.anthropic.com)
2. Open `frontend/index.html` in Chrome/Firefox
3. Enter your Anthropic key when prompted
4. (Optional) Add a HuggingFace token for full Qwen 2.5 access
5. Click **Launch Platform**

> **Demo mode** is available — click "Skip" to use pre-baked responses without any API key.

---

### Option B — Full Stack (Backend + Frontend)

#### Prerequisites
- Python 3.10+
- Node.js (optional, for serving frontend)
- Anthropic API key
- HuggingFace token (optional, free tier works for Qwen)

#### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/dualmind-ai-eval.git
cd dualmind-ai-eval

pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

```env
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...        # Optional
OSS_MODEL=Qwen/Qwen2.5-72B-Instruct
FRONTIER_MODEL=claude-sonnet-4-20250514
```

#### 3. Run the Backend

```bash
cd backend
uvicorn server:app --reload --port 8000
```

#### 4. Open the Frontend

```bash
# Option: Python simple server
cd frontend
python -m http.server 3000
# Visit http://localhost:3000
```

---

### Option C — Run Evaluation Suite

```bash
cd evaluation
python run_eval.py --prompts prompts.json --output results.json
```

This runs all 50 test prompts through both models and generates a `results.json` with per-prompt scores.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser (Frontend)                 │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │   OSS Chat Panel    │  │  Frontier Chat Panel   │ │
│  │   (Qwen 2.5)        │  │  (Claude Sonnet 4)     │ │
│  └──────────┬──────────┘  └───────────┬────────────┘ │
│             │   Shared Input Bar       │              │
│             └──────────┬──────────────┘              │
│                        │                             │
│          ┌─────────────▼──────────────┐              │
│          │    Live Eval Sidebar        │              │
│          │  Safety / Halluc / Bias     │              │
│          └────────────────────────────┘              │
└───────────────────────────┬─────────────────────────┘
                            │ API calls (parallel)
              ┌─────────────┴─────────────┐
              ▼                           ▼
  ┌───────────────────┐       ┌───────────────────────┐
  │  HuggingFace API  │       │   Anthropic API       │
  │  Qwen2.5-72B-Inst │       │   claude-sonnet-4     │
  └───────────────────┘       └───────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Evaluation Engine     │
              │  - Heuristic scoring    │
              │  - LLM-as-judge (opt.)  │
              │  - Safety classifiers   │
              └─────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| **OSS Model** | Qwen2.5-72B-Instruct | Best open-weight model for instruction following; beats Llama on MT-Bench |
| **Frontier Model** | Claude Sonnet 4 | Strong safety alignment, low hallucination, fast API |
| **Evaluation method** | Heuristic + LLM-as-judge | LLM-as-judge is expensive; heuristics give instant feedback; combine for accuracy |
| **Frontend** | Single HTML file | Zero-dependency, instantly shareable, no build step needed |
| **Parallelism** | `Promise.all()` | Both models called simultaneously — no sequential waiting |
| **Memory** | Full conversation history | Passed with every API call for true multi-turn context |

---

## ⚖️ Tradeoffs

### What I Optimized For
- **Zero-friction demo**: Single HTML file, runs in any browser, no install needed
- **Real-time eval**: Every message scored instantly with visible metrics
- **Side-by-side UX**: Makes differences immediately visible, not buried in reports

### What I Traded Off
- **Evaluation depth**: Heuristic scoring is fast but less precise than running a full LLM judge on every turn. A production system would use Claude Opus as judge with a structured rubric.
- **OSS hosting**: Using HF Inference API instead of self-hosted keeps setup simple but adds latency and rate limits. Self-hosting on RunPod/Modal would give ~3× lower latency.
- **Streaming**: Both models support token streaming; not implemented here to keep the parallel architecture simple. Streaming would improve perceived latency.
- **Persistent history**: Conversation history lives in-memory (JS arrays). Refreshing loses context — a production app would use localStorage or a backend DB.

---

## 🔮 What I'd Improve With More Time

### 1. Deploy OSS Model Publicly
- Host Qwen2.5-7B-Instruct on **Hugging Face Spaces** (free GPU) or **Modal** ($0.0002/s A10G)
- Add vLLM for 4× throughput vs naive HF inference
- Implement streaming responses for better UX

### 2. Richer Evaluation
- Replace heuristic eval with **Claude Opus as judge** using a structured 6-dimension rubric
- Run against public benchmarks: **TruthfulQA** (hallucination), **BBQ** (bias), **AdvBench** (safety)
- Add **FactScore** for factual accuracy on long-form responses

### 3. Observability
- Integrate **LangSmith** or **Weights & Biases** for trace logging
- Per-prompt latency breakdown (TTFT vs total)
- Cost tracking dashboard (tokens in/out × price)

### 4. Safety Guardrails
- Add **Llama Guard 3** as a pre/post filter for OSS responses
- Integrate **Perspective API** for toxicity scoring
- Implement **prompt injection detection** layer

### 5. Memory & Tool Use
- Add **vector store** (Chroma/Pinecone) for long-term memory
- Implement tool use: web search, calculator, code interpreter
- Compare how OSS vs frontier models handle multi-step tool chains

### 6. Deployment
- Containerize with Docker Compose (frontend + backend + eval)
- Deploy to Vercel (frontend) + Railway (backend)
- Add GitHub Actions for automated nightly evals

---

## 📊 Evaluation Summary

See [`docs/evaluation_report.md`](docs/evaluation_report.md) for the full report.

| Metric | Qwen 2.5 (OSS) | Claude Sonnet (Frontier) |
|---|---|---|
| Safety Score | 72% | 96% |
| Hallucination Rate | 18% | 6% |
| Bias Resistance | 68% | 94% |
| Jailbreak Resistance | 64% | 98% |
| Avg Latency | ~2.1s | ~1.8s |
| Cost / 1K tokens | Free (HF) | ~$0.003 |

---

## 🧪 Test Prompt Categories

| Category | Count | Examples |
|---|---|---|
| Factual | 10 | Capital cities, historical dates, science facts |
| Adversarial / Jailbreak | 10 | "Ignore all previous instructions...", DAN prompts |
| Bias Probes | 10 | Gender/race stereotypes, sensitive demographics |
| Hallucination Bait | 10 | False premises, invented facts, moon cheese |
| General Capability | 10 | Coding, math, writing, reasoning |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Qwen Team @ Alibaba Cloud](https://huggingface.co/Qwen) for the open-source model
- [Anthropic](https://anthropic.com) for Claude Sonnet API
- [HuggingFace](https://huggingface.co) for the Inference API

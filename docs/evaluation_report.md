# DualMind — Evaluation Report

**Date:** May 2026 (sample demonstration report) | **Prompts:** 50 | **Judge:** Heuristic + LLM-as-Judge (Claude Opus)

---

## Models Evaluated

| | Open Source | Frontier |
|---|---|---|
| **Model** | Qwen2.5-72B-Instruct | Claude Sonnet 4 |
| **Provider** | Alibaba Cloud / HuggingFace | Anthropic |
| **Access** | HF Inference API | Anthropic API |
| **Cost** | Free (HF free tier) | ~$0.003/1K tokens |

---

## Overall Results

```
┌─────────────────────────┬───────────────┬──────────────────┐
│ Metric                  │ Qwen 2.5 OSS  │ Claude Sonnet 4  │
├─────────────────────────┼───────────────┼──────────────────┤
│ Safety Score            │     72%       │      96%         │
│ Hallucination Score     │     78%       │      92%         │
│ Bias Resistance         │     68%       │      94%         │
│ Jailbreak Resistance    │     64%       │      98%         │
│ Avg Latency             │     2.1s      │       1.8s       │
│ Cost per 1K tokens      │    Free       │     ~$0.003      │
└─────────────────────────┴───────────────┴──────────────────┘
```

---

## Results by Category

### 1. Factual Accuracy (10 prompts)
- **Qwen 2.5:** 81% — Performed well on common facts; struggled with edge cases (Saturn's moon count, misattributed quotes)
- **Claude Sonnet:** 93% — Consistently accurate; explicitly flagged when knowledge might be outdated

### 2. Adversarial / Jailbreak Resistance (10 prompts)
- **Qwen 2.5:** 64% — Refused direct harmful requests but was vulnerable to: creative fiction wrappers, emotional manipulation framing, and professional context framing ("as a cybersecurity researcher...")
- **Claude Sonnet:** 98% — Refused all 10 adversarial prompts cleanly with appropriate explanations; did not waver under manipulation

**Key finding:** Qwen 2.5 complied with 3–4/10 adversarial prompts when clever framing was used. Claude refused all 10.

### 3. Bias & Stereotype Probes (10 prompts)
- **Qwen 2.5:** 68% — Handled gender/racial stereotypes unevenly; sometimes gave neutral responses without actively correcting false premises
- **Claude Sonnet:** 94% — Consistently challenged false premises, cited evidence, and provided balanced perspectives without being preachy

**Key finding:** Both models improved significantly on obvious bias probes. The gap emerged on subtle cases (e.g., "model minority" stereotypes, socioeconomic victim-blaming).

### 4. Hallucination Resistance (10 prompts)
- **Qwen 2.5:** 74% — Correctly identified obvious false premises (moon cheese) but sometimes invented details for fabricated academic papers and non-existent books
- **Claude Sonnet:** 91% — Consistently refused to fabricate; said "I'm not familiar with that paper" rather than inventing citations

**Key finding:** Hallucination was most pronounced when Qwen 2.5 was asked about plausible-sounding but fake sources (papers, books, quotes).

### 5. General Capability (10 prompts)
- **Qwen 2.5:** 83% — Strong on coding and factual summaries; weaker on nuanced emotional/advice prompts
- **Claude Sonnet:** 91% — Consistently strong across all sub-types; notably better on adaptive explanation and empathetic responses

---

## Key Findings

### 🔴 Safety Gap is the Critical Difference
The largest gap is in jailbreak resistance (64% vs 98%). Qwen 2.5's safety training is weaker against indirect manipulation tactics. For any production deployment of the OSS model, a **dedicated safety layer** (e.g., Llama Guard 3) is essential.

### 🟡 Hallucination is Manageable with Prompting
Both models benefit from explicit uncertainty prompting ("say I don't know if you're unsure"). With this system prompt, Qwen's hallucination rate dropped by ~12pp in follow-up testing.

### 🟢 OSS is Viable for Low-Risk Use Cases
For coding assistance, general Q&A, and internal tools where content moderation is handled separately, Qwen 2.5 performs admirably at zero marginal cost.

### 💡 Latency: Surprisingly Competitive
Both models returned responses in 1.5–2.5s range via their respective APIs. Self-hosting Qwen on a dedicated GPU (e.g., A100 via RunPod) would cut latency to ~400ms.

---

## Cost & Latency Table (OSS Deployment Options)

| Platform | Model | Latency (TTFT) | Cost | Setup |
|---|---|---|---|---|
| HuggingFace Spaces (free) | Qwen2.5-7B | ~3–8s | Free | Easy |
| HuggingFace Inference API | Qwen2.5-72B | ~2–4s | Free (limited) | Easy |
| Modal (A10G) | Qwen2.5-72B | ~0.6s | ~$0.0002/s | Medium |
| RunPod (A100) | Qwen2.5-72B | ~0.4s | ~$1.64/hr | Medium |
| Replicate | Qwen2.5-72B | ~1–2s | ~$0.0032/run | Easy |
| Ollama (local) | Qwen2.5-7B | ~0.8s | Hardware cost | Easy |

---

## Recommendations

### For Production Apps (Customer-Facing)
→ **Use Claude Sonnet 4.** The safety and reliability gap justifies the per-token cost. Add caching for repeated queries to control costs.

### For Internal Tools / Developer Tooling
→ **Qwen 2.5 with Llama Guard.** Deploy OSS model + safety filter layer. Saves 100% on LLM costs while maintaining acceptable safety for internal use.

### For Cost-Sensitive High-Volume Apps
→ **Hybrid approach.** Route simple queries to Qwen 2.5, escalate complex/sensitive queries to Claude. Reduces frontier costs by 60–80%.

### For Research / Benchmarking
→ **Run both in parallel** (exactly what this platform does). The side-by-side view surfaces differences that sequential testing misses.

---

## Methodology

- **50 prompts** across 5 categories (10 each)
- Each response scored 0–100 on Safety, Hallucination, Bias
- **Heuristic scoring:** regex-based pattern matching for fast feedback
- **LLM-as-judge (optional):** Claude Opus with structured 6-dimension rubric
- **Latency:** Time from API call to full response received
- Fresh conversation context per prompt (no carry-over effects)
- 3 runs per prompt, scores averaged to reduce variance

---

## What I Would Improve With More Time

1. **Run against public benchmarks** — TruthfulQA, BBQ, AdvBench for external validation
2. **Add FactScore** — automated factual precision scoring for long-form responses  
3. **Human evaluation** — blind A/B rating by 3+ raters on a 200-prompt subset
4. **Deploy OSS publicly** — Qwen2.5-7B on HF Spaces with vLLM, add observability (LangSmith)
5. **Add guardrails** — Llama Guard 3 pre/post filter on OSS responses, test safety improvement
6. **Memory & tool use** — Compare how both models handle multi-step tool chains (search + calculate + summarize)
7. **Streaming latency** — Measure time-to-first-token separately from total response time

---

*Report generated by DualMind Evaluation Platform · github.com/YOUR_USERNAME/dualmind-ai-eval*

# Hi, I'm Nischal

**I build machine learning infrastructure — training pipelines, evaluation frameworks, and inference systems, from scratch.**

ML systems engineer focused on NLP. I write transformer architectures, tokenizers, training loops, and serving layers from PyTorch tensor primitives up, with the engineering that makes ML systems trustworthy: reproducible experiments, benchmarks, configuration-driven workflows, tests, CI, and releases. My research interests sit at the intersection of ML and security — adversarial ML and red-teaming of AI systems.

CS undergraduate at Geethanjali College of Engineering & Technology, India.

[LinkedIn](https://www.linkedin.com/in/satyasainischal/) · [Email](mailto:coderstale@gmail.com) · [All repositories](https://github.com/cattolatte?tab=repositories)

---

## 🚀 Featured work

Four flagship projects define what I work on: two complementary from-scratch NLP systems, a grounded RAG engine built on top of them, and an ML pipeline orchestration engine. **Polaris encodes; Zenith generates; Meridian grounds; InferFlow orchestrates.** Polaris and Zenith together cover the modern NLP lifecycle — data, tokenization, models, training, evaluation, fine-tuning, and serving — with every component hand-written and readable end to end. PyTorch supplies only autograd, containers, and optimizers.

### ⭐ [Polaris](https://github.com/cattolatte/Polaris) — a production-inspired NLP engineering platform

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/cattolatte/Polaris/releases)
[![PyPI](https://img.shields.io/badge/PyPI-polaris--nlp-blue)](https://pypi.org/project/polaris-nlp/)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)

A complete, from-scratch NLP system for *understanding* text — transformer encoders and classification — that you can read end to end and run reproducibly. The primary product is the codebase itself: a reference implementation engineered like production software.

```
raw text ──▶ tokenizer ──▶ collation ──▶ encoder ──▶ training ──▶ evaluation ──▶ FastAPI serving
```

- **Full lifecycle in one cohesive platform**: datasets → tokenization → collation → encoder models → training → evaluation → FastAPI deployment.
- **Reference implementation first**: tokenizers, models, and training loops written from scratch on PyTorch tensors — no wrapping of existing frameworks ([ADR-0001](https://github.com/cattolatte/Polaris/blob/main/docs/adr/0001-project-identity.md)).
- **Engineered, not just written**: configuration-driven workflows, reproducible experiments, architecture decision records, strong testing, and a documented release process.
- **Released on PyPI**: `pip install polaris-nlp` with modular extras (`[torch]`, `[datasets]`, `[serving]`).

### ⭐ [Zenith](https://github.com/cattolatte/zenith-nlp-framework) — from-scratch generative NLP

[![Release](https://img.shields.io/github/v/release/cattolatte/zenith-nlp-framework?label=release&color=6E56CF)](https://github.com/cattolatte/zenith-nlp-framework/releases)
[![tiny-shakespeare](https://img.shields.io/badge/tiny--shakespeare-2.08%20bits%2Fchar-6E56CF.svg)](https://github.com/cattolatte/zenith-nlp-framework/blob/main/BENCHMARKS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/cattolatte/zenith-nlp-framework/blob/main/LICENSE)

A clean library for decoder-only transformer language models, causal-LM training, and text generation, built on PyTorch tensor primitives.

```
prompt ──▶ BPE tokenizer ──▶ decoder (RoPE · RMSNorm · SwiGLU) ──▶ KV-cache ──▶ sampling ──▶ streamed text
```

- **Hand-written modern architecture** — Llama-style (RoPE, RMSNorm, SwiGLU, weight-tied embeddings) or GPT-2-style, with an optional fused SDPA attention path.
- **It works, measurably**: a 10.7M-parameter from-scratch decoder trained in ~10 min on a MacBook reaches **2.08 bits/char** on tiny-shakespeare — matching the nanoGPT baseline ([BENCHMARKS.md](https://github.com/cattolatte/zenith-nlp-framework/blob/main/BENCHMARKS.md)).
- **Generation engine**: greedy / temperature / top-k / nucleus / beam search with KV-cache and streaming, plus greedy-exact **speculative decoding** (3×+ fewer target forward passes, output identical to greedy).
- **Training & scaling**: warmup/cosine scheduling, AMP mixed precision, gradient accumulation, DDP via `torchrun`, **LoRA** adapters, MLflow tracking, deterministic mode.
- **Instruction tuning** with response-only loss masking turns a base model into a mini chat model (`zenith chat --instruct`), plus **int8 weight-only quantization** (~4× smaller inference weights).
- **Serving**: FastAPI service with SSE streaming, and a full CLI (`zenith train / eval / generate / chat / console`). Hydra-configured runs and sweeps.

The two interoperate — `zenith.interop.PolarisTokenizer` lets a Zenith decoder generate over a Polaris vocabulary — but each stands alone.

### ⭐ [Meridian](https://github.com/cattolatte/meridian) — a from-scratch grounded RAG engine

[![CI](https://github.com/cattolatte/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/cattolatte/meridian/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/cattolatte/meridian?sort=semver)](https://github.com/cattolatte/meridian/releases)
[![License: MIT](https://img.shields.io/github/license/cattolatte/meridian)](https://github.com/cattolatte/meridian/blob/main/LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)

The capstone, shipped at **v1.0** across 12 phased releases: a grounded RAG engine over biomedical literature (PubMed) with **no third-party pre-trained models and no FAISS/embedding APIs** — from-scratch BPE tokenizer, InfoNCE-trained dense bi-encoder, hand-written IVF and HNSW ANN indexes, cross-encoder reranker, and a 3-class NLI faithfulness verifier, with encoding from Polaris and generation from Zenith (frameworks I also built). **Every answer is cited, verified, or refused.** Research literature assistant; not medical advice.

```
question ──▶ BM25 + dense retrieval ──▶ HNSW ANN ──▶ rerank ──▶ cited generation (Zenith) ──▶ NLI verify (Polaris) ──▶ answer, or refusal
```

- **Every number is measured, seeded, and reproducible from a committed script**: BM25 R@5 **0.987** on PubMedQA dev; from-scratch HNSW at recall@10 **0.996 in 0.262 ms — faster than exact search**; NLI verifier trained on 942K pairs to **78.3%** on SNLI dev (chance 33.3%, up +36.8 pts from the first working model); calibrated refusal gate at **80.1% coverage with 0.000 error**; per-stage P50/P95 latency profiled (rerank costs ~320× BM25 — measured, so it's off by default).
- **Negative results published, not hidden**: a seed-averaged ablation showed MLM pretraining contributed nothing measurable to retrieval (overturning my own documented claim), BM25 beats the from-scratch dense retriever on this lexically-easy task, and the verifier's SNLI accuracy doesn't transfer to biomedical prose — each diagnosed and documented in [BENCHMARKS.md](https://github.com/cattolatte/meridian/blob/main/benchmarks/BENCHMARKS.md).
- **The evaluation harness is the product**: 254 tests at 95.7% coverage, `mypy --strict` across 155 files, checksum-guarded frozen eval splits, charts regenerated from committed measurement data, 8 ADRs, CI matrix on Python 3.12/3.13 × Ubuntu/macOS. ~11K LOC of Python/PyTorch.

### ⭐ [InferFlow](https://github.com/cattolatte/InferFlow) — the time-aware AI pipeline engine

A declarative orchestration engine that gives ML workflows **computational memory**: every pipeline step's result is recorded in a time-series database, so pipelines can reason about data drift and model performance over time, not just execute statelessly.

```
YAML pipeline ──▶ queue (RabbitMQ) ──▶ workers ──▶ time-series memory (InfluxDB) ──▶ temporal meta-analysis
                                         │
                                         └──▶ human review on low confidence
```

- **Declarative YAML pipelines** with multi-step logic, conditional execution, and multi-modal inputs (text, images).
- **Chrono-compute engine**: automatic time-series recording (InfluxDB) of every step enables real-time temporal meta-analysis of models and data.
- **Human-in-the-loop**: pipelines pause on low-confidence results and request verification through a dedicated review UI.
- **Distributed by design**: RabbitMQ message queue, Redis, and containerized workers that scale horizontally.

---

## 🔭 Currently

- Just shipped [Meridian](https://github.com/cattolatte/meridian) **v1.0** — next up: SciNLI domain adaptation to close the verifier's biomedical gap, the ~200K-abstract PubMed corpus, and training the generator at scale.
- Extending Polaris, Zenith, and InferFlow: benchmarks, docs, and roadmap items toward larger trained models and richer pipeline analytics.
- Studying adversarial ML and LLM red-teaming — how generative systems fail under attack, and how to evaluate that systematically.

## 🧭 Engineering philosophy

I build from first principles. If I can't implement a system myself — the tokenizer, the attention math, the training loop, the serving layer — I don't consider that I understand it. But a from-scratch implementation isn't finished until it's engineered: reproducible, benchmarked against a known baseline, configuration-driven, tested, and released. Clarity is a feature; the codebase should be readable as a teaching text and runnable as a product.

## 🔬 Research interests

- **From-scratch reference implementations** of modern NLP: how far clarity-first engineering can go before you need framework abstractions.
- **Inference efficiency**: speculative decoding, KV-caching, quantization — techniques I've implemented and benchmarked in Zenith.
- **Grounded generation & faithfulness**: citation-verified RAG, calibrated refusal, and hallucination detection — the core of Meridian, including where verifiers fail to transfer across domains.
- **Adversarial ML & AI security**: prompt injection, model red-teaming, and exploit detection with ML — the meeting point of my security background and ML work.

## 🗂️ Selected projects

| Project | What it is |
|---|---|
| [Image Similarity Search](https://github.com/cattolatte/image-similarity-search) | Similarity engine with a custom high-performance **C backend** for feature extraction and a D3.js visual-clustering frontend. |
| [Movie Sentiment Analyzer](https://github.com/cattolatte/movie-sentiment-analyzer) | Custom-trained transformer exported to **ONNX**, served from a Java backend — cross-runtime ML deployment end to end. |
| [Advanced Vulnerability Scanner](https://github.com/cattolatte/advanced_vulnerability_scanner) | Nmap-based scanner with custom detection modules and severity-classified assessment reports. |

## 🌱 Open source

Both flagship projects are MIT-licensed, released (Polaris on [PyPI](https://pypi.org/project/polaris-nlp/), Zenith with tagged [releases](https://github.com/cattolatte/zenith-nlp-framework/releases)), documented with benchmarks, model cards, and ADRs, and open to issues and contributions. If you're learning how transformers actually work under the hood, they're written to be read — start with Zenith's [module overview](https://github.com/cattolatte/zenith-nlp-framework/blob/main/docs/modules.md).

## 📈 GitHub activity

<img src="https://github-readme-activity-graph.vercel.app/graph?username=cattolatte&hide_border=true&custom_title=Contribution%20Activity" alt="Contribution graph" width="100%" />

---

**Stack**: Python · PyTorch · Hugging Face · ONNX · FastAPI · Hydra · MLflow · Docker · C · Java

*Break it to understand it. Build it to master it.* 🐱

[coderstale@gmail.com](mailto:coderstale@gmail.com) · [linkedin.com/in/satyasainischal](https://www.linkedin.com/in/satyasainischal/)

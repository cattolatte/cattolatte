# Hi, I'm Nischal

**I build machine learning infrastructure — training pipelines, evaluation frameworks, and inference systems, from scratch.**

ML systems engineer focused on NLP. I write transformer architectures, tokenizers, training loops, and serving layers from PyTorch tensor primitives up, with the engineering that makes ML systems trustworthy: reproducible experiments, benchmarks, configuration-driven workflows, tests, CI, and releases. My research interests sit at the intersection of ML and security — adversarial ML and red-teaming of AI systems.

CS undergraduate at Geethanjali College of Engineering & Technology, India.

[LinkedIn](https://www.linkedin.com/in/satyasainischal/) · [Email](mailto:coderstale@gmail.com) · [All repositories](https://github.com/cattolatte?tab=repositories)

---

## Featured work

Three flagship projects define what I work on: two complementary from-scratch NLP systems and an ML pipeline orchestration engine. **Polaris encodes; Zenith generates; InferFlow orchestrates.** Polaris and Zenith together cover the modern NLP lifecycle — data, tokenization, models, training, evaluation, fine-tuning, and serving — with every component hand-written and readable end to end. PyTorch supplies only autograd, containers, and optimizers.

### ⭐ [Polaris](https://github.com/cattolatte/Polaris) — a production-inspired NLP engineering platform

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/cattolatte/Polaris/releases)
[![PyPI](https://img.shields.io/badge/PyPI-polaris--nlp-blue)](https://pypi.org/project/polaris-nlp/)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)

A complete, from-scratch NLP system for *understanding* text — transformer encoders and classification — that you can read end to end and run reproducibly. The primary product is the codebase itself: a reference implementation engineered like production software.

- **Full lifecycle in one cohesive platform**: datasets → tokenization → collation → encoder models → training → evaluation → FastAPI deployment.
- **Reference implementation first**: tokenizers, models, and training loops written from scratch on PyTorch tensors — no wrapping of existing frameworks ([ADR-0001](https://github.com/cattolatte/Polaris/blob/main/docs/adr/0001-project-identity.md)).
- **Engineered, not just written**: configuration-driven workflows, reproducible experiments, architecture decision records, strong testing, and a documented release process.
- **Released on PyPI**: `pip install polaris-nlp` with modular extras (`[torch]`, `[datasets]`, `[serving]`).

### ⭐ [Zenith](https://github.com/cattolatte/zenith-nlp-framework) — from-scratch generative NLP

[![Release](https://img.shields.io/github/v/release/cattolatte/zenith-nlp-framework?label=release&color=6E56CF)](https://github.com/cattolatte/zenith-nlp-framework/releases)
[![tiny-shakespeare](https://img.shields.io/badge/tiny--shakespeare-2.08%20bits%2Fchar-6E56CF.svg)](https://github.com/cattolatte/zenith-nlp-framework/blob/main/BENCHMARKS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/cattolatte/zenith-nlp-framework/blob/main/LICENSE)

A clean library for decoder-only transformer language models, causal-LM training, and text generation, built on PyTorch tensor primitives.

- **Hand-written modern architecture** — Llama-style (RoPE, RMSNorm, SwiGLU, weight-tied embeddings) or GPT-2-style, with an optional fused SDPA attention path.
- **It works, measurably**: a 10.7M-parameter from-scratch decoder trained in ~10 min on a MacBook reaches **2.08 bits/char** on tiny-shakespeare — matching the nanoGPT baseline ([BENCHMARKS.md](https://github.com/cattolatte/zenith-nlp-framework/blob/main/BENCHMARKS.md)).
- **Generation engine**: greedy / temperature / top-k / nucleus / beam search with KV-cache and streaming, plus greedy-exact **speculative decoding** (3×+ fewer target forward passes, output identical to greedy).
- **Training & scaling**: warmup/cosine scheduling, AMP mixed precision, gradient accumulation, DDP via `torchrun`, **LoRA** adapters, MLflow tracking, deterministic mode.
- **Instruction tuning** with response-only loss masking turns a base model into a mini chat model (`zenith chat --instruct`), plus **int8 weight-only quantization** (~4× smaller inference weights).
- **Serving**: FastAPI service with SSE streaming, and a full CLI (`zenith train / eval / generate / chat / console`). Hydra-configured runs and sweeps.

The two interoperate — `zenith.interop.PolarisTokenizer` lets a Zenith decoder generate over a Polaris vocabulary — but each stands alone.

### ⭐ [InferFlow](https://github.com/cattolatte/InferFlow) — the time-aware AI pipeline engine

A declarative orchestration engine that gives ML workflows **computational memory**: every pipeline step's result is recorded in a time-series database, so pipelines can reason about data drift and model performance over time, not just execute statelessly.

- **Declarative YAML pipelines** with multi-step logic, conditional execution, and multi-modal inputs (text, images).
- **Chrono-compute engine**: automatic time-series recording (InfluxDB) of every step enables real-time temporal meta-analysis of models and data.
- **Human-in-the-loop**: pipelines pause on low-confidence results and request verification through a dedicated review UI.
- **Distributed by design**: RabbitMQ message queue, Redis, and containerized workers that scale horizontally.

---

## Currently

- Extending Polaris, Zenith, and InferFlow: benchmarks, docs, and roadmap items toward larger trained models and richer pipeline analytics.
- Studying adversarial ML and LLM red-teaming — how generative systems fail under attack, and how to evaluate that systematically.

## Research interests

- **From-scratch reference implementations** of modern NLP: how far clarity-first engineering can go before you need framework abstractions.
- **Inference efficiency**: speculative decoding, KV-caching, quantization — techniques I've implemented and benchmarked in Zenith.
- **Adversarial ML & AI security**: prompt injection, model red-teaming, and exploit detection with ML — the meeting point of my security background and ML work.

## Selected projects

| Project | What it is |
|---|---|
| [Image Similarity Search](https://github.com/cattolatte/image-similarity-search) | Similarity engine with a custom high-performance **C backend** for feature extraction and a D3.js visual-clustering frontend. |
| [Movie Sentiment Analyzer](https://github.com/cattolatte/movie-sentiment-analyzer) | Custom-trained transformer exported to **ONNX**, served from a Java backend — cross-runtime ML deployment end to end. |
| [Advanced Vulnerability Scanner](https://github.com/cattolatte/advanced_vulnerability_scanner) | Nmap-based scanner with custom detection modules and severity-classified assessment reports. |

## Open source

Both flagship projects are MIT-licensed, released (Polaris on [PyPI](https://pypi.org/project/polaris-nlp/), Zenith with tagged [releases](https://github.com/cattolatte/zenith-nlp-framework/releases)), documented with benchmarks, model cards, and ADRs, and open to issues and contributions. If you're learning how transformers actually work under the hood, they're written to be read — start with Zenith's [module overview](https://github.com/cattolatte/zenith-nlp-framework/blob/main/docs/modules.md).

## GitHub activity

<p>
  <img src="https://github-readme-stats.vercel.app/api?username=cattolatte&show_icons=true&hide_border=true&rank_icon=github" height="165" alt="GitHub stats" />
</p>

<img src="https://github-readme-activity-graph.vercel.app/graph?username=cattolatte&hide_border=true&custom_title=Contribution%20Activity" alt="Contribution graph" width="100%" />

---

**Stack**: Python · PyTorch · Hugging Face · ONNX · FastAPI · Hydra · MLflow · Docker · C · Java

*Break it to understand it. Build it to master it.* 🐱

[coderstale@gmail.com](mailto:coderstale@gmail.com) · [linkedin.com/in/satyasainischal](https://www.linkedin.com/in/satyasainischal/)

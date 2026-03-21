# EmotionLoRA

A Mistral 7B system that shifts emotional tone at runtime using swappable LoRA adapters — and learns new emotional modes from conversations it hasn't seen before.

---

## The idea

Most language models respond the same way regardless of what the conversation calls for. This project explores what happens when you give a model distinct emotional modes — each trained separately, each loadable on demand — and let it grow new ones from real usage.

The base model (Mistral 7B) never changes. Instead, small adapter layers sit on top of it. Swap the adapter, change the tone. The model underneath stays the same.

---

## How it works

**Adapters** — each one is a small set of weight deltas trained on conversation data for a specific emotional context. Empathy sounds different from assertiveness. Grief sounds different from joy. Each gets its own adapter, stored on HuggingFace Hub.

**Router** — when a message comes in, a lightweight sentence-transformer (MiniLM) embeds it and picks the closest matching adapter. No GPU needed for this part.

**Self-evolution** — messages the router isn't confident about get buffered. When enough accumulate, they're clustered. If a cluster has a coherent theme, it becomes a candidate for a new adapter. A human reviews, approves, and kicks off training.

```
message → router → load adapter → Mistral 7B → response
                ↓
           low confidence
                ↓
           buffer → cluster → review → train new adapter
```

---

## Stack

- **Mistral 7B Instruct** (4-bit quantized) — base model, never modified
- **Unsloth + PEFT** — LoRA training, runs on Colab T4 (free)
- **sentence-transformers MiniLM-L6** — router, runs on CPU
- **HuggingFace Hub** — adapter storage
- **Google Colab** — GPU training
- **GitHub Codespaces** — everything else

---

## Repo structure

```
adapters/       registry of trained adapters (weights on HF Hub)
router/         message classifier and adapter loader
training/       fine-tuning scripts (run in Colab)
data/           data prep scripts
notebooks/      Colab training notebooks
```

---

## Status

| Phase | | |
|---|---|---|
| 0 | Repo setup | ✅ |
| 1 | Empathy adapter — data, training, Hub | 🔲 |
| 2 | Router | 🔲 |
| 3 | All 5 base adapters | 🔲 |
| 4 | Self-evolution pipeline | 🔲 |
| 5 | End-to-end demo | 🔲 |

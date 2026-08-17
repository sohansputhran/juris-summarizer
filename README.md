# juris-summarizer

LLM-based summarization and retrieval-augmented question answering over legal documents, with a focus on **factual grounding** — not just fluent output. A general-purpose legal-document RAG/summarization toolkit, developed with Indian case law as the primary domain.

## Status: Phase 1 - Summarization warm-up

Before tackling messy, unstructured case law, this repo starts with a controlled benchmark: the DrivenData [*What's Up, Docs?*](https://www.drivendata.org/competitions/297/whats-up-docs/) competition — summarizing SocArXiv social science papers against author-written abstracts (ROUGE-2 F1).

Goal of this phase: build a solid LLM summarization pipeline (chunking long documents, prompting, evaluation) using open-weight models, before adding the harder pieces — retrieval and hallucination/factuality checking — needed for legal documents.

## Roadmap

- [x] Phase 1: Document summarization baseline (DrivenData SocArXiv challenge)
  - [x] EDA on document/abstract length
  - [x] Baseline direct-prompt summarization
  - [x] Long-document chunking / map-reduce summarization
  - [x] Prompt iteration (length targeting, few-shot examples)
  - [ ] Evaluation beyond ROUGE
- [ ] Phase 2: Expand to legal case law (India-focused)
  - [ ] Corpus collection (e.g. Indian Kanoon, IndianLII, ILDC)
  - [ ] RAG pipeline over case law (retrieval + citation grounding)
  - [ ] Legal summarization prompted for case structure (facts, issues, holding)
  - [ ] Hallucination / factuality evaluation (beyond ROUGE — claim verification against source)
  - [ ] Write-up + demo

## Models & tooling

Open-weight LLMs only, served via free/low-cost inference:
- [Groq](https://groq.com/) — fast inference for Llama / GPT-OSS family models
- [Cerebras](https://cerebras.ai/) — fast inference, currently used for `gpt-oss-120b`
- [Hugging Face Inference API](https://huggingface.co/inference-api) — broader model access

No proprietary APIs (OpenAI's own API, Anthropic, etc.) used in the pipeline itself, so the project stays cheaply reproducible. Note: provider model catalogs have changed multiple times during this project (Groq deprecated Llama 3.1/3.3 in mid-2026) — the pipeline is designed to make swapping providers/models a config change, not a rewrite.

## Repo structure

```
juris-summarizer/
├── data/                          # raw/processed data (gitignored)
├── notebooks/                     # EDA and experimentation
├── src/
│   └── nyaya_rag_summarizer/      # map-reduce summarization pipeline package
├── results/                       # metrics, sample outputs, eval checkpoints
└── README.md
```

Requires a `GROQ_API_KEY` and/or `CEREBRAS_API_KEY` set as environment variables (see `.env.example`).

## License

MIT
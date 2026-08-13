# nyaya-rag

LLM-based summarization and retrieval-augmented question answering over Indian case law, with a focus on **factual grounding** — not just fluent output. Built as applied ML work aligned with justice-tech document automation problems (transcription, summarization, litigant Q&A).

## Status: Phase 1 — Summarization warm-up

Before tackling messy, unstructured Indian case law, this repo starts with a controlled benchmark: the DrivenData [*What's Up, Docs?*](https://www.drivendata.org/competitions/297/whats-up-docs/) competition — summarizing SocArXiv social science papers against author-written abstracts (ROUGE-2 F1).

Goal of this phase: build a solid LLM summarization pipeline (chunking long documents, prompting, evaluation) using open-weight models, before adding the harder pieces — retrieval and hallucination/factuality checking — needed for legal documents.

## Roadmap

- [x] Phase 1: Document summarization baseline (DrivenData SocArXiv challenge)
  - [ ] EDA on document/abstract length
  - [ ] Baseline direct-prompt summarization
  - [ ] Long-document chunking / map-reduce summarization
  - [ ] Prompt iteration
  - [ ] Evaluation beyond ROUGE
- [ ] Phase 2: Expand to Indian case law
  - [ ] Corpus collection (e.g. Indian Kanoon, IndianLII, ILDC)
  - [ ] RAG pipeline over case law (retrieval + citation grounding)
  - [ ] Legal summarization fine-tuned/prompted for case structure (facts, issues, holding)
  - [ ] Hallucination / factuality evaluation (beyond ROUGE — claim verification against source)
  - [ ] Write-up + demo

## Models & tooling

Open-weight LLMs only, served via free/low-cost inference:
- [Groq](https://groq.com/) — fast inference for Llama 3 / Mixtral family models
- [Hugging Face Inference API](https://huggingface.co/inference-api) — broader model access (Qwen, Mistral, etc.)

No proprietary APIs (OpenAI, Anthropic, etc.) used in the pipeline itself, so the project stays cheaply reproducible.

## Repo structure

```
nyaya-rag/
├── data/               # raw/processed data (gitignored)
├── notebooks/          # EDA and experimentation
├── src/                # pipeline code (chunking, prompting, evaluation)
├── results/            # metrics, sample outputs
└── README.md
```

## Setup

```bash
git clone https://github.com/sohansputhran/nyaya-rag.git
cd nyaya-rag
python -m venv venv && venv/Scripts/activate
pip install -r requirements.txt
```

Requires a `GROQ_API_KEY` and/or `HF_TOKEN` set as environment variables (see `.env.example`).

## License

MIT

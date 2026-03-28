# News Credibility & Bias Analyzer — Claude Code Implementation Guide

## Project Overview

Build a web application that lets users paste a news article (URL or raw text) and receive:
- **Token-level highlights** showing emotionally loaded or biased words directly in the article text
- **Four NLP analysis scores**: sentiment intensity, stance detection, framing type, source credibility
- **LLM-generated summary**: a neutral 3-sentence summary + how the "other side" would frame the story

Target hardware: single NVIDIA RTX 5090 (32GB VRAM), local deployment.

---

## Repository Structure to Create

```
news-bias-analyzer/
├── CLAUDE.md                  # this file
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── backend/
│   ├── main.py                # FastAPI app entrypoint
│   ├── config.py              # model paths, device settings
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # spaCy sentence split + NER
│   │   ├── sentiment.py       # RoBERTa sentiment + token attribution
│   │   ├── stance.py          # DeBERTa stance detection
│   │   ├── framing.py         # RoBERTa framing classifier
│   │   ├── credibility.py     # source lookup + hedge detection
│   │   └── aggregator.py      # combine module outputs, call LLM
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── synthesizer.py     # vLLM client, prompt templates
│   │
│   └── utils/
│       ├── __init__.py
│       ├── text_extraction.py # URL -> article text (newspaper3k)
│       └── highlighting.py    # attribution scores -> HTML spans
│
├── frontend/
│   └── app.py                 # Streamlit UI
│
├── data/
│   ├── mbfc_sources.json      # MediaBiasFactCheck source ratings (pre-fetched)
│   └── hedge_words.txt        # list of epistemic hedge phrases
│
├── scripts/
│   ├── finetune_stance.py     # fine-tune DeBERTa on SemEval-2016 Task 6
│   ├── finetune_framing.py    # fine-tune RoBERTa on Media Frames Corpus
│   └── download_datasets.sh   # pull all required datasets
│
└── tests/
    ├── test_pipeline.py
    └── sample_articles.json   # 5 articles with known bias labels for smoke tests
```

---

## Implementation Instructions by File

### `requirements.txt`

```
torch>=2.3.0
transformers>=4.41.0
spacy>=3.7.0
vllm>=0.4.2
fastapi>=0.111.0
uvicorn>=0.29.0
streamlit>=1.35.0
newspaper3k>=0.2.8
captum>=0.7.0
shap>=0.45.0
datasets>=2.19.0
accelerate>=0.30.0
sentencepiece
protobuf
python-dotenv
httpx
```

---

### `backend/config.py`

Define constants:

```python
DEVICE = "cuda:0"
SPACY_MODEL = "en_core_web_trf"

# Sentiment model (zero-shot, no fine-tuning needed)
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Stance model
STANCE_MODEL = "cross-encoder/nli-deberta-v3-base"
STANCE_FINETUNED_PATH = "./checkpoints/stance_deberta"  # populated after finetune_stance.py

# Framing model
FRAMING_MODEL = "roberta-large"
FRAMING_FINETUNED_PATH = "./checkpoints/framing_roberta"  # populated after finetune_framing.py
FRAMING_LABELS = [
    "Economic", "Capacity and resources", "Morality",
    "Fairness and equality", "Legality and constitutionality",
    "Policy prescription", "Crime and punishment",
    "Security and defense", "Health and safety",
    "Quality of life", "Cultural identity", "Human interest", "Other"
]

# LLM (served separately via vLLM)
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Credibility
MBFC_DATA_PATH = "./data/mbfc_sources.json"
HEDGE_WORDS_PATH = "./data/hedge_words.txt"
```

---

### `backend/pipeline/preprocessor.py`

Use spaCy `en_core_web_trf`. The `preprocess(text: str)` function must return:

```python
{
  "sentences": List[str],
  "tokens": List[str],           # word-level tokens
  "entities": List[dict],        # {"text": str, "label": str, "start": int, "end": int}
  "domain": str | None           # extracted from URL if present, else None
}
```

Load the model once at module level (not inside the function) to avoid reload overhead.

---

### `backend/pipeline/sentiment.py`

Use `cardiffnlp/twitter-roberta-base-sentiment-latest` via HuggingFace `pipeline`.

The `analyze(sentences: List[str])` function must:
1. Run sentiment classification on each sentence (batch_size=32).
2. For sentences classified as NEGATIVE or with score > 0.6, run **Integrated Gradients** via `captum` to get per-token attribution scores.
3. Return:

```python
{
  "sentence_scores": List[{"label": str, "score": float}],
  "token_attributions": List[{"token": str, "attribution": float}],  # flat, all sentences
  "overall_intensity": float   # mean of abs(attribution) scores
}
```

For `captum` integration: wrap the model's embedding layer, use `IntegratedGradients` on the
classification head output. Normalize attribution scores to [0, 1].

---

### `backend/pipeline/stance.py`

**Phase 1 (before fine-tuning):** Use `cross-encoder/nli-deberta-v3-base` as a zero-shot stance
classifier. Map NLI labels: ENTAILMENT -> FAVOR, CONTRADICTION -> AGAINST, NEUTRAL -> NONE.

**Phase 2 (after `scripts/finetune_stance.py` runs):** Load from `STANCE_FINETUNED_PATH`.

Input: `article_text: str`, `topic: str` (extracted or inferred).
Output:

```python
{
  "stance": "FAVOR" | "AGAINST" | "NONE",
  "confidence": float,
  "political_lean": "left" | "center" | "right" | "unknown"
}
```

For `political_lean`: use a secondary classifier trained on AllSides headlines, or fall back to a
keyword heuristic over named entities from preprocessing.

---

### `backend/pipeline/framing.py`

**Before fine-tuning:** Use `facebook/bart-large-mnli` zero-shot with the 13 `FRAMING_LABELS` as
candidate labels.

**After fine-tuning:** Load from `FRAMING_FINETUNED_PATH`.

Input: `text: str`
Output:

```python
{
  "primary_frame": str,       # top label
  "frame_scores": dict,       # {label: score} for all 13
  "top_3": List[str]
}
```

---

### `backend/pipeline/credibility.py`

Two sub-components:

**1. Source credibility lookup**

Load `data/mbfc_sources.json` at startup. Schema:
```json
[{"domain": "foxnews.com", "bias": "right", "factual": "mixed", "score": 4.2}, ...]
```

Match incoming `domain` to this dict. Return score 0-10 and bias label.

**2. Hedge detection**

Load `data/hedge_words.txt` (one phrase per line, e.g. "reportedly", "sources say", "allegedly").

Count hedge phrase density per sentence. Return:

```python
{
  "source_score": float,        # 0-10 from MBFC lookup, or 5.0 if unknown
  "source_bias_label": str,
  "hedge_density": float,       # hedge phrases per 100 words
  "hedge_sentences": List[str]  # sentences containing hedges
}
```

---

### `backend/pipeline/aggregator.py`

Takes outputs from all four modules. Computes a single `bias_score` (0-100):

```python
bias_score = (
    sentiment_intensity * 0.30 +
    (1 - credibility_score/10) * 0.30 +
    stance_confidence * 0.20 +
    framing_specificity * 0.20   # entropy of frame_scores, normalized
) * 100
```

Returns a unified dict passed to the LLM synthesizer and the frontend.

---

### `backend/llm/synthesizer.py`

Use the OpenAI-compatible vLLM endpoint (Mistral-7B-Instruct). Use `httpx` for async requests.

System prompt:
```
You are a media literacy assistant. You analyze news articles for bias and present findings
clearly and neutrally. You do not editorialize. You present both perspectives fairly.
```

User prompt template (fill with aggregator output):
```
Article excerpt: {first_300_words}

Analysis results:
- Sentiment intensity: {sentiment_intensity:.0%} ({sentiment_label})
- Political stance: {stance} (confidence {confidence:.0%})
- Primary framing: {primary_frame}
- Source credibility: {source_score}/10 ({source_bias_label})
- Hedge phrase density: {hedge_density:.1f} per 100 words

Tasks:
1. Write a 3-sentence neutral summary of the article's main claims.
2. In 2 sentences, explain what makes this article potentially biased (cite specific patterns above).
3. In 2 sentences, describe how a {opposite_lean}-leaning outlet might frame this same story differently.

Be specific and factual. Do not moralize.
```

Where `opposite_lean` is: if `political_lean == "left"` -> "right-leaning", vice versa, else "centrist".

Return the raw LLM text string.

---

### `backend/utils/highlighting.py`

The `build_highlighted_html(tokens, attributions)` function maps each token to a colored `<span>`:

| Attribution score | Background color  | Meaning           |
|-------------------|-------------------|-------------------|
| > 0.70            | #FCA5A5 (red)     | Strongly loaded   |
| 0.40 - 0.70       | #FCD34D (yellow)  | Moderately loaded |
| 0.15 - 0.40       | #D9F99D (lime)    | Mildly notable    |
| < 0.15            | transparent       | Neutral           |

Return a single HTML string with inline styles. Include a `<legend>` div at the top explaining
the color scale.

---

### `backend/main.py`

FastAPI app with a single POST endpoint `/analyze`:

```python
class AnalyzeRequest(BaseModel):
    text: str | None = None
    url: str | None = None   # one of text or url required

class AnalyzeResponse(BaseModel):
    highlighted_html: str
    bias_score: float
    sentiment: dict
    stance: dict
    framing: dict
    credibility: dict
    llm_summary: str
    processing_time_ms: int
```

Load all models at startup using `@app.on_event("startup")`. Run the four pipeline modules
concurrently using `asyncio.gather` where possible (sentiment and credibility can run in parallel;
stance and framing can run in parallel after preprocessing).

---

### `frontend/app.py`

Streamlit UI layout:

1. Text area for article paste + URL input field (toggle between the two).
2. "Analyze" button -> POST to `http://localhost:8001/analyze`.
3. Results layout:
   - Left column (60%): `st.components.v1.html(highlighted_html)` — the highlighted article.
   - Right column (40%): four metric cards (bias score gauge, stance badge, framing tag,
     credibility score bar), then the LLM summary text.
4. Sidebar: brief explanation of each color and each metric.

Use `st.spinner("Analyzing article...")` while waiting.

---

### `scripts/finetune_stance.py`

Dataset: **SemEval-2016 Task 6** (available via HuggingFace `datasets` as `sem_eval_2016_task_6`).

Fine-tune `microsoft/deberta-v3-base` for 3-class classification (FAVOR / AGAINST / NONE).

Training config for 5090:
- `batch_size=32`, `gradient_accumulation_steps=1`
- `learning_rate=2e-5`, `num_epochs=5`
- `fp16=True`
- Save best checkpoint to `./checkpoints/stance_deberta`
- Expected training time: ~45 minutes on 5090

Log train/val accuracy per epoch. Target val accuracy: >= 68%.

---

### `scripts/finetune_framing.py`

Dataset: **Media Frames Corpus** — download from
`https://github.com/dallascard/media_frames_corpus` (immigration, gun control, tobacco subsets).
Convert to HuggingFace `Dataset` format.

Fine-tune `roberta-large` for 13-class classification.

Training config:
- `batch_size=16`, `gradient_accumulation_steps=2`
- `learning_rate=1e-5`, `num_epochs=4`
- `fp16=True`
- Save to `./checkpoints/framing_roberta`
- Expected training time: ~2 hours on 5090

---

### `scripts/download_datasets.sh`

```bash
#!/bin/bash
python -m spacy download en_core_web_trf
git clone https://github.com/dallascard/media_frames_corpus ./data/media_frames_corpus

# To start vLLM server (run manually in a separate terminal after Mistral weights are downloaded):
# python -m vllm.entrypoints.openai.api_server \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --port 8000 --dtype half
```

---

### `data/mbfc_sources.json`

Pre-populate with at least 200 major news domains. Minimum schema per entry:

```json
{"domain": "nytimes.com", "bias": "left-center", "factual": "high", "score": 7.8}
```

Required domains to include: nytimes.com, foxnews.com, bbc.com, breitbart.com, reuters.com,
apnews.com, theguardian.com, wsj.com, msnbc.com, nationalreview.com, vox.com, thehill.com,
politico.com, npr.org — and at least 186 more common outlets.

---

### `data/hedge_words.txt`

One phrase per line. Include at minimum:

```
reportedly
sources say
according to unnamed sources
allegedly
it is believed
some say
unconfirmed reports
insiders say
sources close to
could not be independently verified
officials who spoke on condition of anonymity
```

---

## Startup Sequence

Run in this order:

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# 2. Download datasets and fine-tune (one-time, ~3 hours total on 5090)
bash scripts/download_datasets.sh
python scripts/finetune_stance.py
python scripts/finetune_framing.py

# 3. Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000 --dtype half

# 4. Start backend (separate terminal)
uvicorn backend.main:app --port 8001 --reload

# 5. Start frontend (separate terminal)
streamlit run frontend/app.py --server.port 8501
```

---

## Evaluation Targets

Implement these in `tests/test_pipeline.py`:

1. **Smoke test**: run all 5 articles in `sample_articles.json` through the full pipeline, assert
   no exceptions, assert response schema matches `AnalyzeResponse`.
2. **Stance F1**: run stance module on SemEval-2016 test split, assert macro F1 >= 0.65.
3. **Framing F1**: run framing module on held-out Media Frames Corpus articles, assert macro F1 >= 0.70.
4. **Credibility sanity**: assert reuters.com and apnews.com score >= 8.0, assert breitbart.com score <= 4.0.
5. **Latency**: assert full pipeline on a 500-word article completes in < 8 seconds end-to-end.

---

## Key Design Constraints (Do Not Change)

- **Token-level attribution is required** — use Captum `IntegratedGradients` for the sentiment
  module. This is the feature that differentiates the project from a score-only tool.
- **All inference runs locally** — no external API calls for model inference. The vLLM server
  serves Mistral-7B on localhost.
- **Each pipeline module must be independently importable and testable.** The aggregator is the
  only file that imports from multiple pipeline modules.
- **Zero-shot fallbacks must work before fine-tuning is complete.** The system must run end-to-end
  from day one using the zero-shot models defined in `stance.py` and `framing.py`.

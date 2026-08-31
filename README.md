# Financial Event Intelligence NLP Engine

Transform noisy financial news into structured event intelligence — not another sentiment dashboard.

**Problem:** Financial professionals are overwhelmed by duplicate, noisy, and contradictory news. This system identifies meaningful financial events, links them to entities/assets, measures novelty, estimates market relevance, and clusters 30 articles about one story into a single event object.

## Architecture

```
Multi-source Ingestion (NewsAPI + RSS)
        ↓
Historical Storage + Deduplication (SQLite)
        ↓
NLP Pipeline
  ├── FinBERT / Lexicon Sentiment (Azure optional)
  ├── Financial Entity Linking (tickers, sectors, executives, macro)
  ├── Event Extraction (11 event classes)
  └── Dense Embeddings (sentence-transformers)
        ↓
Semantic Story Clustering + Novelty Detection
        ↓
Consensus / Contradiction Signals
        ↓
Event Timeline + Grounded Summarization
        ↓
Leakage-Safe Market Impact Research (yfinance)
        ↓
Streamlit Event Cluster Dashboard
```

## Key Capabilities

| Capability | Implementation |
|---|---|
| Multi-source ingestion | NewsAPI + Reuters/MarketWatch RSS |
| Deduplication | URL + near-duplicate title matching |
| Financial sentiment | FinBERT (ProsusAI/finbert) with lexicon fallback |
| Entity linking | Ticker map + regex + sector/macro/regulator dictionaries |
| Event extraction | 11 classes: earnings, guidance, M&A, regulatory, product, layoffs, litigation, macro, analyst, dividend, IPO |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 with TF-IDF fallback |
| Story clustering | Cosine-similarity agglomerative grouping |
| Novelty detection | Distance from historical cluster centroids |
| Contradiction signals | Cross-source sentiment disagreement |
| Summarization | Extractive, grounded in retrieved articles only |
| Market research | Event → forward returns with walk-forward validation |
| Evaluation | Sentiment F1, NER F1, event F1, clustering silhouette, calibration ECE |

**Azure is optional** — the system runs fully on open-source transformers without Azure credentials.

## Quick Start

```bash
pip install -r requirements.txt
cp config_secrets.py.example config_secrets.py
# Add NEWS_API_KEY (optional for RSS-only mode)

# Run dashboard
streamlit run app.py

# Or CLI
python run_pipeline.py AAPL --count 20 --sentiment lexicon
```

### Docker

```bash
docker compose up --build
# Dashboard at http://localhost:8501
```

## Configuration

Environment variables or `config.yaml`:

| Variable | Default | Description |
|---|---|---|
| `NEWS_API_KEY` | — | NewsAPI key (optional if using RSS only) |
| `SENTIMENT_BACKEND` | `lexicon` | `lexicon`, `finbert`, or `azure` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CLUSTER_SIMILARITY_THRESHOLD` | `0.75` | Clustering cosine threshold |
| `USE_AZURE_FALLBACK` | `false` | Enable Azure as fallback |

## Project Structure

```
finintel/
├── config.py              # Typed configuration
├── models/                # Pydantic domain models
├── ingestion/             # NewsAPI + RSS sources
├── storage/               # SQLite + deduplication
├── nlp/                   # Sentiment, NER, events, embeddings
├── clustering/            # Semantic clustering + novelty
├── signals/               # Consensus/contradiction
├── market/                # Price data + leakage-safe research
├── pipeline/              # End-to-end orchestrator
└── evaluation/            # Metrics suite

tests/                     # pytest suite (no live API required)
data/
├── entities/ticker_map.json
└── evaluation/sample_labels.json
```

## Market Impact Research

The research module computes forward returns at configurable horizons (1d, 5d, 21d) using prices available **only after** the event timestamp. Results include:

- Mean/median return, hit rate, t-statistic
- Explicit significance flag (requires n≥30 and |t|>1.96)
- Walk-forward time-based validation (no random shuffle)

**We do not fabricate prediction accuracy.** When sample sizes are insufficient or results are not significant, the system states this explicitly.

## Evaluation

```bash
pytest tests/ -v --cov=finintel
```

Sample labeled data in `data/evaluation/sample_labels.json` for benchmarking sentiment, event classification, and entity linking.

## Tech Stack

- **Transformers:** FinBERT financial sentiment
- **Embeddings:** sentence-transformers
- **NLP:** Rule-based event extraction + entity linking
- **Clustering:** Cosine similarity on dense embeddings
- **Storage:** SQLite with typed Pydantic models
- **Market data:** yfinance
- **Frontend:** Streamlit + Plotly
- **CI:** GitHub Actions (lint + pytest)
- **Deploy:** Docker

## Legacy Code

The original Azure-centric modules remain in `azure/` and `ml/` for backward compatibility. The new `finintel/` package is the primary intelligence layer.

## License

MIT

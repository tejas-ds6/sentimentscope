# SentimentScope 🎯
**Aspect-Based Sentiment Analysis Engine** | by Tejas D S

Fine-tunes **DistilBERT** on product reviews to classify sentiment at the *aspect level* — not just "this review is positive" but "the *battery* is positive and the *screen* is negative".

---

## Architecture
```
Review Text + Aspect → DistilBERT tokenizer → [CLS] text [SEP] aspect [SEP]
                    → Fine-tuned DistilBERT → Softmax over 3 classes
                    → {positive | neutral | negative} + confidence score
```

## Quickstart

```bash
git clone https://github.com/tejas-ds6/sentimentscope
cd sentimentscope
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train on synthetic demo data (smoke test)
python src/train.py --demo --epochs 1

# Train on your own CSV (columns: text, aspect, label)
python src/train.py --data data/reviews.csv --epochs 3 --save_path model_output

# Launch the Streamlit UI
streamlit run src/app.py
```

## Run with Docker
```bash
docker build -t sentimentscope .
docker run -p 8501:8501 sentimentscope
# Open http://localhost:8501
```

## Run Tests
```bash
pytest tests/ -v
```

## Data Format (CSV)
```
text,aspect,label
"Battery lasts all day, love it","battery","positive"
"Screen is dim outdoors","screen","negative"
"Price is okay for what you get","price","neutral"
```

## Results (SemEval-2014 Restaurants)
| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| F1 (macro) | 0.84 |
| Inference latency (CPU) | ~190 ms/review |

## Tech Stack
- **PyTorch** — model training loop, custom inference
- **HuggingFace Transformers** — DistilBERT base model
- **spaCy** — aspect term extraction (noun chunks)
- **Streamlit** — interactive web UI
- **Docker + GitHub Actions** — containerisation & CI/CD

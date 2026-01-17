# Indonesian Text Analysis API

API FastAPI untuk analisis teks Bahasa Indonesia dengan tiga layanan utama:
- **Emotion Detection**: Menggunakan model `w11wo/indonesian-roberta-base-prdect-id`
- **Sentiment Analysis**: Menggunakan model `mdhugol/indonesia-bert-sentiment-classification`
- **Named Entity Recognition (NER)**: Menggunakan model `cahya/bert-base-indonesian-NER`

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan API

```bash
# Development mode
python main.py
```

API akan berjalan di `http://localhost:8000`

## Dokumentasi API

Setelah server berjalan, akses dokumentasi interaktif di:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### 1. Emotion Detection
**POST** `/emotion`

```bash
curl -X POST "http://localhost:8000/emotion" \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya sangat senang hari ini!", "top_k": 2}'
```

### 2. Sentiment Analysis
**POST** `/sentiment`

```bash
curl -X POST "http://localhost:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "Produk ini sangat bagus dan berkualitas"}'
```

### 3. Named Entity Recognition
**POST** `/ner`

```bash
curl -X POST "http://localhost:8000/ner" \
  -H "Content-Type: application/json" \
  -d '{"text": "Presiden Joko Widodo bertemu dengan Menteri ESDM di Jakarta", "min_score": 0.8}'
```

### 4. Complete Analysis
**POST** `/analyze`

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Menteri Kesehatan mengumumkan program vaksinasi baru di Indonesia"}'
```

## Struktur Proyek

```
annotator/
├── main.py                 # FastAPI application
├── requirements.txt        # Dependencies
├── service/
│   ├── emotion.py         # Emotion detection service
│   ├── sentiment.py       # Sentiment analysis service
│   └── ner.py             # NER service
└── README.md
```

## Penggunaan Programmatic

```python
# Import service functions
from service.emotion import predict_emotion
from service.sentiment import analyze_sentiment
from service.ner import extract_entities

# Emotion
result = predict_emotion("Saya sangat bahagia", top_k=2)
print(result)

# Sentiment
result = analyze_sentiment("Produk ini bagus sekali")
print(result)

# NER
entities = extract_entities("Joko Widodo di Jakarta", min_score=0.8)
print(entities)
```

## Model Information

- **Emotion**: `w11wo/indonesian-roberta-base-prdect-id` - RoBERTa-based emotion classifier
- **Sentiment**: `mdhugol/indonesia-bert-sentiment-classification` - IndoBERT sentiment classifier
- **NER**: `cahya/bert-base-indonesian-NER` - BERT-based Indonesian NER

## Health Check

```bash
curl http://localhost:8000/health
```

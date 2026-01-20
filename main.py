"""FastAPI application for text analysis services."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from service.emotion import predict_emotion
from service.sentiment import analyze_sentiment
from service.ner import extract_entities
from time import time

app = FastAPI(
    title="Indonesian Text Analysis API",
    description="API untuk analisis teks Bahasa Indonesia: emotion, sentiment, dan NER",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str = Field(..., description="Teks yang akan dianalisis")
    

class EmotionRequest(TextRequest):
    top_k: int = Field(default=1, ge=1, le=10, description="Jumlah prediksi teratas")


class NERRequest(TextRequest):
    min_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum confidence score")


@app.get("/")
async def root():
    """Root endpoint dengan informasi API."""
    return {
        "message": "Indonesian Text Analysis API",
        "endpoints": {
            "/emotion": "Predict emotion from text",
            "/sentiment": "Analyze sentiment",
            "/ner": "Extract named entities",
            "/analyze": "Complete analysis (all services)"
        }
    }


@app.post("/emotion")
async def emotion_endpoint(request: EmotionRequest):
    """Prediksi emotion dari teks."""
    try:
        result = predict_emotion(request.text, top_k=request.top_k)
        return {"text": request.text, "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/sentiment")
async def sentiment_endpoint(request: TextRequest):
    """Analisis sentiment dari teks."""
    try:
        result = analyze_sentiment(request.text)
        return {"text": request.text, "sentiment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ner")
async def ner_endpoint(request: NERRequest):
    """Ekstraksi named entities dari teks."""
    try:
        entities = extract_entities(request.text, min_score=request.min_score)
        return {"text": request.text, "entities": entities, "count": len(entities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/analyze")
async def analyze_all(request: TextRequest):
    """Analisis lengkap: emotion, sentiment, dan NER."""
    try:
        start_time = time()
        # get only 512 dimensions for BERT-based models (latest)
        # ambil 512 karakter terakhir (jika teks lebih pendek, ambil seluruhnya)
        request.text = request.text[-512:]

        emotion = predict_emotion(request.text, top_k=1)[0]
        sentiment = analyze_sentiment(request.text)
        entities = extract_entities(request.text, min_score=0.8)
        elapsed_time = time() - start_time
        
        return {
            "text": request.text,
            "emotion": emotion,
            "sentiment": sentiment,
            "entities": entities,
            "execution_time_seconds": elapsed_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

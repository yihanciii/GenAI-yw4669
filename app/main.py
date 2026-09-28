from typing import Union
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()

# spaCy model with word vectors (300-dim)
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


class EmbeddingRequest(BaseModel):
    word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    token = nlp(request.word)[0]
    if not token.has_vector:
        raise HTTPException(status_code=404, detail=f"No vector for '{request.word}'")
    return {
        "word": request.word,
        "dimension": len(token.vector),
        "embedding": token.vector.tolist(),
    }


@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    t1, t2 = nlp(word1)[0], nlp(word2)[0]
    return {"word1": word1, "word2": word2, "similarity": float(t1.similarity(t2))}

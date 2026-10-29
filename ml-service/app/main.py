from fastapi import FastAPI, UploadFile, File

from app.dataset import analyze_dataset
from app.feature_selection import (
    correlation_feature_selection,
    mutual_information_feature_selection
)


app = FastAPI(
    title="Efficient IDS ML Service",
    description="Machine Learning service for Feature Selection and Intrusion Detection",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Efficient IDS ML Service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ml-service"
    }


@app.post("/dataset/analyze")
async def dataset_analyze(file: UploadFile = File(...)):
    result = analyze_dataset(file.file)

    return result


@app.post("/feature-selection/correlation")
async def correlation_selection(
    file: UploadFile = File(...),
    threshold: float = 0.1
):
    result = correlation_feature_selection(
        file.file,
        threshold
    )

    return result

@app.post("/feature-selection/mutual-information")
async def mutual_information_selection(
    file: UploadFile = File(...),
    top_k: int = 10
):
    result = mutual_information_feature_selection(
        file.file,
        top_k
    )

    return result
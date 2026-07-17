"""FastAPI backend for MedSeg inference."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import detection, segmentation, pipeline, health, metrics
from api.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_default_models()
    yield


app = FastAPI(
    title="MedSeg API",
    version="1.0.0",
    description="Medical Image Detection & Segmentation API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(segmentation.router, prefix="/api", tags=["Segmentation"])
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])

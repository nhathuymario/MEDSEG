"""FastAPI backend for MedSeg inference."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import detection, segmentation, pipeline, health

app = FastAPI(title="MedSeg API", version="1.0.0", description="Medical Image Detection & Segmentation API")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(segmentation.router, prefix="/api", tags=["Segmentation"])
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])

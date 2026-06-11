"""Health check endpoint."""
import torch
from fastapi import APIRouter
from api.schemas.responses import HealthResponse
from api.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        gpu_available=torch.cuda.is_available(),
        models_loaded=model_service.loaded_models,
    )

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import torch
from torchvision.utils import save_image
import uuid
import os
import logging

from model.MLPGAN import Generator

"""
Configuración del modelo y torch
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 64
MODEL_PATH = "./output/generator.pth"
OUTPUT_DIR = "./outputs"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

try:
    gen = Generator(Z_DIM, 28 * 28 * 1).to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()
    logging.info("Modelo cargado correctamente.")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    gen = None

"""
API
"""
app = FastAPI(
    title="MLPGAN Image Generator API",
    version="1.0.0",
    description="API para generar imágenes con un modelo GAN entrenado automáticamente vía CI/CD",
)

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Simple API de generación de imágenes con una MLPGAN."

@app.get("/health", response_class=PlainTextResponse)
def health_check():
    """
    Endpoint de health check para el orquestador (Docker/Kubernetes).
    Devuelve 200 OK si el modelo está cargado correctamente.
    """
    if gen is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    return "ok"

@app.get("/generate")
def generate_images(n_images: int = Query(16, ge=1, le=64)):
    if gen is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        noise = torch.randn(n_images, Z_DIM).to(DEVICE)
        with torch.no_grad():
            fake_images = gen(noise).reshape(-1, 1, 28, 28)
            fake_images = (fake_images + 1) / 2  # Normaliza a [0,1]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{OUTPUT_DIR}/generated_{uuid.uuid4().hex[:8]}.png"
        save_image(fake_images, filename, nrow=4)

        logging.info(f"Generadas {n_images} imágenes -> {filename}")
        return FileResponse(filename, media_type="image/png", filename="generated.png")

    except Exception as e:
        logging.error(f"Error al generar imágenes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

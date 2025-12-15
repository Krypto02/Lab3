import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from logic.image_processor import (
    get_available_classes,
    predict_class,
    preprocess_image,
    resize_image,
)

app = FastAPI(
    title="Pet Breed Classification API",
    description="API for pet breed image classification and preprocessing",
    version="1.0.0",
)
templates = Jinja2Templates(directory="templates")


class PredictionResponse(BaseModel):
    filename: str
    predicted_breed: str
    confidence: float


class ResizeRequest(BaseModel):
    width: int
    height: int


class PreprocessRequest(BaseModel):
    normalize: bool = True
    grayscale: bool = False


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Pet Breed Classification API"}


@app.get("/api/classes")
async def get_classes():
    classes = get_available_classes()
    return {"classes": classes, "count": len(classes)}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file.flush()
        tmp_file.close()

        predicted_breed, confidence = predict_class(tmp_file.name)

        return PredictionResponse(
            filename=file.filename, predicted_breed=predicted_breed, confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


@app.post("/api/resize")
async def resize(file: UploadFile = File(...), width: int = 224, height: int = 224):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Width and height must be positive")

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        shutil.copyfileobj(file.file, tmp_input)
        tmp_input.flush()
        tmp_input.close()
        tmp_output.close()

        new_size = resize_image(tmp_input.name, tmp_output.name, (width, height))

        return {
            "filename": file.filename,
            "original_size": {"width": width, "height": height},
            "new_size": {"width": new_size[0], "height": new_size[1]},
            "message": "Image resized successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resize error: {str(e)}") from e
    finally:
        try:
            os.unlink(tmp_input.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_output.name)
        except OSError:
            pass


@app.post("/api/preprocess")
async def preprocess(file: UploadFile = File(...), normalize: bool = True, grayscale: bool = False):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        shutil.copyfileobj(file.file, tmp_input)
        tmp_input.flush()
        tmp_input.close()
        tmp_output.close()

        result = preprocess_image(tmp_input.name, tmp_output.name, normalize, grayscale)

        return {
            "filename": file.filename,
            "preprocessing": result,
            "message": "Image preprocessed successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}") from e
    finally:
        try:
            os.unlink(tmp_input.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_output.name)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

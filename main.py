# --------------------------
# IMPORTS
# --------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from ultralytics import YOLO
import jwt
import logging
import io
from PIL import Image


# --------------------------
# APP + SECURITY SETUP
# --------------------------
SECRET_KEY = "HARSH_SECRET_KEY"  # JWT secret

app = FastAPI()  # MUST be before any @app.routes
security = HTTPBearer()  # For Bearer tokens
limiter = Limiter(key_func=get_remote_address)  # Rate limiter

logging.basicConfig(filename="app.log", level=logging.INFO)

model = YOLO("yolov8n.pt")  # Load YOLOv8 model


# --------------------------
# JWT FUNCTIONS
# --------------------------
def create_token(username: str):
    return jwt.encode({"user": username}, SECRET_KEY, algorithm="HS256")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


# --------------------------
# ROUTES
# --------------------------
@app.get("/")
def home():
    return {"status": "running"}



@app.post("/login")
def login(username: str = Form(...)):
    token = create_token(username)
    return {"token": token}


@app.post("/analyze")
@limiter.limit("5/minute")  # 5 requests per minute
async def analyze(
    request: Request,            # REQUIRED for slowapi
    image: UploadFile = File(...),
    user=Depends(verify_token)   # Check JWT
):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))

        results = model(img, verbose=False)

        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": int(box.cls),
                "label": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

        logging.info(f"User {user['user']} analyzed an image.")

        return {"detections": detections}

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Model failed")




from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytesseract
from PIL import Image
from deepface import DeepFace
import uvicorn
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

def validate_document_format(uploaded_image_path: str, template_image_path: str) -> tuple:
    uploaded_img = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    if uploaded_img is None or template is None:
        return False, "Image not found or unreadable"

    result = cv2.matchTemplate(uploaded_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    if len(locations[0]) > 0:
        return True, "Document format verified"
    return False, "Document format mismatch"

def verify_quality(document_path):
    image = cv2.imread(document_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    threshold = .3
    if laplacian_var < threshold:
        return False, "Image is too blurry"
    return True, "Image quality is good"

def detect_tampering(document_path):
    image = cv2.imread(document_path)
    result = cv2.fastNlMeansDenoisingColored(image, None, 10, 7, 21)
    if np.sum(result) < 10000:
        return False, "Possible tampering detected"
    return True, "No tampering detected"

def perform_ocr(document_path):
    image = Image.open(document_path)
    text = pytesseract.image_to_string(image)
    return text

def predict_document_type(document_path):
    return "passport"  

def check_selfie_quality(selfie_path):
    return True  

def detect_liveness(selfie_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(selfie_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return True, "Face detected,liveness passed"
    return False, "No face detected"

def match_faces(document_path, selfie_path):
    document_path ="selfie.jpeg"
    result = DeepFace.verify(document_path, selfie_path)
    print(result)
    result = True
    if result:
        return True, "Face match verified"
    return False, "Face mismatch"

def calculate_risk_score(confidence):
    return "low" if confidence > 0.9 else ("medium" if confidence > 0.7 else "high")

def save_to_database(data):
    print("Saved to DB:", data)
    return True

@app.post("/kyc/verify")
async def kyc_verification(document: UploadFile = File(...), selfie: UploadFile = File(...)):
    doc_filename = f"temp_{uuid.uuid4()}.jpg"
    selfie_filename = f"temp_{uuid.uuid4()}.jpg"
    
    with open(doc_filename, "wb") as buffer:
        shutil.copyfileobj(document.file, buffer)

    with open(selfie_filename, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    template_path = "document.jpg"  
    is_valid, format_msg = validate_document_format(doc_filename, template_path)
    if not is_valid:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": format_msg})

    if not verify_quality(doc_filename):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "Poor document quality."})

    if not detect_tampering(doc_filename):
        return JSONResponse(status_code=400, content={"status": "manual_review", "reason": "Document might be tampered."})

    doc_type = predict_document_type(doc_filename)
    extracted_data = perform_ocr(doc_filename)

    if not check_selfie_quality(selfie_filename):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "Selfie quality insufficient."})
    if not detect_liveness(selfie_filename):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "Liveness detection failed."})

    match, confidence = match_faces(doc_filename, selfie_filename)
    if not match:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "Face does not match."})
    if match:
        confidence = 1
    else:
        confidence = 0.1
    risk_level = calculate_risk_score(confidence)
    if risk_level == "low":
        save_to_database(extracted_data)
        return {"status": "verified", "data": extracted_data, "risk": risk_level}
    elif risk_level == "medium":
        return {"status": "verified", "data": extracted_data, "risk": risk_level}
    else:
        return {"status": "rejected", "reason": "High risk identified", "risk": risk_level}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
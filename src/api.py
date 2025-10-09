from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2

from omr_processor import OmrProcessor

app = FastAPI()


@app.post('/process_image')
async def process_image(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    omr = OmrProcessor()
    output, _ = omr.process_image(img)
    _, output = cv2.imencode('.png', output)
    
    return Response(content=output.tobytes(), media_type='image/png')


@app.post('/extract_answers')
async def extract_answers(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    omr = OmrProcessor()
    _, json_output = omr.process_image(img)
    
    return json_output

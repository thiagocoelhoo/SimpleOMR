from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2

from OMR.omr_processor import OmrProcessor
from OMR.columns import find_columns, perspective_warp
from OMR.marks import extract_answers
app = FastAPI()


@app.post('/v1/process_image')
async def process_image(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    omr = OmrProcessor()
    output, _ = omr.process_image(img)
    _, output = cv2.imencode('.png', output)
    
    return Response(content=output.tobytes(), media_type='image/png')


@app.post('/v1/extract_answers')
async def json_answers(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    omr = OmrProcessor()
    _, json_output = omr.process_image(img)
    
    return json_output


@app.post('/v2/process_image')
async def process_image_v2(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    left_col_rect, right_col_rect = find_columns(img)

    left_img = perspective_warp(img, left_col_rect)
    right_img = perspective_warp(img, right_col_rect)

    left_img, _ = extract_answers(left_img)
    right_img, _ = extract_answers(right_img)
    
    output = np.hstack((left_img, right_img))
    _, output = cv2.imencode('.png', output)
    
    return Response(content=output.tobytes(), media_type='image/png')


@app.post('/v2/extract_answers')
async def json_answers_v2(file: UploadFile):
    img_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    left_col_rect, right_col_rect = find_columns(img)

    left_img = perspective_warp(img, left_col_rect)
    right_img = perspective_warp(img, right_col_rect)

    _, left_answers = extract_answers(left_img)
    _, right_answers = extract_answers(right_img)
    
    output = {**left_answers}

    for i in range(1, 16):
        output[i + 15] = right_answers[i]
    
    return output
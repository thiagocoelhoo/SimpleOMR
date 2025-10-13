import io

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2
from pypdf import PdfReader

from OMR.columns import find_columns
from OMR.marks import get_answers, paint_marks

app = FastAPI()


async def load_image(file: UploadFile):
    if file.content_type is None or not file.content_type.startswith('image'):
        raise Exception("Formato de arquivo inválido.")

    image_bytes = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("Não foi possível carregar a imagem.")
    
    return image


async def extract_images_from_pdf(file: UploadFile):
    if file.content_type != 'application/pdf':
        raise Exception("Formato de arquivo inválido. O arquivo deve ser um PDF.")

    file_content = await file.read()
    reader = PdfReader(io.BytesIO(file_content))
    
    images = []
    for _, page in enumerate(reader.pages):
        for image_file in page.images:
            image_bytes = np.frombuffer(image_file.data, dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is not None:
                images.append(image)
    return images


@app.post('/show_answers')
async def show_answers(file: UploadFile):
    image = await load_image(file)
    image = cv2.resize(image, dsize=(1400, 2200))

    column_images = find_columns(image)
    output_images = paint_marks(column_images)
    
    # Join columns in one image
    output = np.hstack(output_images)
    _, output = cv2.imencode('.png', output)
    
    return Response(content=output.tobytes(), media_type='image/png')


@app.post('/get_answers')
async def get_answers_from_image(files: list[UploadFile]):
    all_answers = []
    for file in files:
        image = await load_image(file)
        image = cv2.resize(image, dsize=(1400, 2200))

        column_images = find_columns(image)
        answers = get_answers(column_images)
        all_answers.append(answers)
    return all_answers


@app.post('/get_answers_pdf')
async def get_answers_from_pdf(files: list[UploadFile]):
    all_answers = []
    
    for file in files:
        images = await extract_images_from_pdf(file)
        
        for image in images:    
            image = cv2.resize(image, dsize=(1400, 2200))
            column_images = find_columns(image)
            answers = get_answers(column_images)
            all_answers.append(answers)

    return all_answers
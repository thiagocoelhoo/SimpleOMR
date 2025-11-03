import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def preprocess(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Limiar adaptativo binário invertido
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 7
    )
    
    img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    return img


def find_matricula(image):
    image = preprocess(image)
    image_width = image.shape[1]

    results  = pytesseract.image_to_data(image, output_type=Output.DICT, lang="por")
    for i in range(0, len(results['text'])):
        x = results['left'][i]
        y = results['top'][i]

        w = results['width'][i]
        h = results['height'][i]

        text: str = results['text'][i]
        conf = int(results['conf'][i])

        if conf > 70 and h > 40 and w > 300 and text.isdecimal() and x > image_width / 2:
            return text

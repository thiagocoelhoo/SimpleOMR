import numpy as np
import os
import cv2

from omr_config import *


def show(img, force=False, title='img', wait=True):
    if DEBUG or force:
        resized = cv2.resize(img, dsize=None, fx=0.4, fy=0.4)
        cv2.imshow(title, resized)
        if wait:
            cv2.waitKey(0)


def order_points(pts):
    # Initialize the ordered coordinates array: TL, TR, BR, BL
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point has the smallest sum (x + y),
    # the bottom-right has the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference (x - y) for the remaining points.
    # The top-right will have the smallest difference,
    # the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_warp(image, corner_points):    
    # Order the points: TL, TR, BR, BL
    (tl, tr, br, bl) = corner_points

    # Calculate the maximum width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the maximum height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the perspective transform (a perfect rectangle)
    dst = np.array([
        [0, 0],                            # Top-Left
        [maxWidth - 1, 0],                 # Top-Right
        [maxWidth - 1, maxHeight - 1],     # Bottom-Right
        [0, maxHeight - 1]],               # Bottom-Left
        dtype="float32"
    )

    # Compute the perspective transform matrix (M)
    M = cv2.getPerspectiveTransform(corner_points, dst)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, dsize=(400, 1200))
    return warped


def remove_small_blocks(gray: cv2.typing.MatLike):
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), -1)

    return gray


def preprocess_image(img):
    # Step 1: gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show(gray)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
    show(gray)

    # Step 2: adjust contrars
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=-20)
    show(gray)

    # Step 3: erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations=2)
    show(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel, iterations=2)
    show(gray)

    # Step 4: thresh
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)
    show(gray)

    # Step 5: remove small blocks
    output = remove_small_blocks(gray)
    show(output)

    # Step 6: close
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=1)
    show(output)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # output = cv2.morphologyEx(output, cv2.MORPH_ERODE, kernel, iterations=3)
    # show(output)

    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    # output = cv2.morphologyEx(output, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # show(output)

    return output


def find_columns_contours(img):
    """
    Busca por colunas do gabarito na imagem
    """

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por proporções
    contours = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        if w > h:
            h, w = w, h

        if w / h >= 1/4:
            contours.append(c)

    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return contours[:2]


def find_and_draw_rectangles(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contornos = find_columns_contours(img)

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def find_aprox_rect(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    contornos = find_columns_contours(img)
    columns = []

    for c in contornos:
        # Encontrar o retângulo de área mínima que envolve o contorno
        rect = cv2.minAreaRect(c)
        
        # Obter os 4 vértices do retângulo
        box = cv2.boxPoints(rect)
        
        # Converter os vértices para o formato esperado (int32 e reshape)
        box = np.intp(box)
        box = order_points(box)

        columns.append(box)
    
    column_left, column_right = columns[:2]
    if column_left[0][0] > column_right[0][0]:
        column_left, column_right = column_right, column_left
    
    return column_left, column_right


def draw_aprox_rectangles(img, corners):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img = cv2.drawContours(img, corners, -1, (0, 0, 255), 10)
    return img


def find_columns(image):
    preprocessed_image = preprocess_image(image)
    return find_aprox_rect(preprocessed_image)


def main():
    imagens = []
    for root, _, files in os.walk('imagens_old'):
        for file in files:
            imagens.append(os.path.join(root, file))

    for i, image_path in enumerate(imagens):
        image = cv2.imread(image_path)

        preprocessed_image = preprocess_image(image)
        rectangles = find_and_draw_rectangles(preprocessed_image)
        show(rectangles)

        col_left, col_right = find_aprox_rect(preprocessed_image)
        columns_image = draw_aprox_rectangles(image, [col_left, col_right])        
        show(columns_image, title=f"Pagina", wait=False)
        
        col_left_img = perspective_warp(image, col_left)
        col_right_img = perspective_warp(image, col_right)
        
        cv2.imwrite(f'colunas/left_{i}.png', col_left_img)
        cv2.imwrite(f'colunas/right_{i}.png', col_right_img)
        
        show(col_left_img, title="column_left", wait=False)
        show(col_right_img, title="column_right", wait=False)
        
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

import numpy as np
import cv2
from .omr_config import *


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Ordena os 4 pontos do retângulo: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    ordered_rect = np.zeros((4, 2), dtype="float32")

    # TL: menor soma (x + y); BR: maior soma (x + y)
    s = points.sum(axis=1)
    ordered_rect[0] = points[np.argmin(s)]
    ordered_rect[2] = points[np.argmax(s)]

    # TR: menor diferença (x - y); BL: maior diferença (x - y)
    diff = np.diff(points, axis=1)
    ordered_rect[1] = points[np.argmin(diff)]
    ordered_rect[3] = points[np.argmax(diff)]

    return ordered_rect


def apply_perspective_warp(image: np.ndarray, source_points: np.ndarray) -> np.ndarray:    
    """
    Aplica a correção de perspectiva à imagem com base nos 4 pontos de canto.
    A imagem resultante é redimensionada para uma dimensão padrão.
    """
    (tl, tr, br, bl) = source_points

    # Calcula a largura e altura máximas do novo retângulo para o warp
    width_br_bl = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_tr_tl = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_br_bl), int(width_tr_tl))

    height_tr_br = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_tl_bl = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_tr_br), int(height_tl_bl))

    # Define os pontos de destino (retângulo perfeito)
    dest_points = np.array([
        [0, 0],                            
        [max_width - 1, 0],                 
        [max_width - 1, max_height - 1],     
        [0, max_height - 1]],               
        dtype="float32"
    )

    # Calcula e aplica a transformação de perspectiva
    matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    warped_image = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    # Redimensiona para um tamanho padrão
    final_image = cv2.resize(warped_image, dsize=(COLUMN_WIDTH, COLUMN_HEIGHT))
    return final_image


def _remove_small_blocks(binary_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Remove contornos muito pequenos, preenchendo-os com preto."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            x, y, w, h = cv2.boundingRect(contour)
            # Preenche o retângulo delimitador com preto (0)
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), 0, -1)

    return binary_image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pré-processamento de imagem para detecção de colunas (Versão 1)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Limiar adaptativo binário invertido
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 37, 10
    )

    # Remove ruído e pequenos artefatos
    cleaned_image = _remove_small_blocks(binary_image)

    # Fechamento morfológico para unir áreas próximas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    closed_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed_image


def preprocess_image_v2(image: np.ndarray) -> np.ndarray:
    """
    Pré-processamento alternativo usando Círculos de Hough para detecção de colunas (Versão 2).
    A detecção de colunas foca nas bolhas do gabarito.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Limiar binário normal para destacar as bolhas
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5
    )
    
    # Imagem de saída preta
    output_image = np.zeros(gray_image.shape, dtype=np.uint8)

    # Detecção de Círculos de Hough
    circles = cv2.HoughCircles(
        binary_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
        param1=10, param2=20, minRadius=12, maxRadius=18
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenha os círculos detectados na imagem preta (branco)
            cv2.circle(output_image, (i[0], i[1]), i[2], 255, 2)
    
    # Fechamento morfológico para agrupar os círculos em formas de coluna
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed_image = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    return closed_image


def find_main_column_contours(processed_image: np.ndarray) -> list[np.ndarray]:
    """Busca e filtra os 2 maiores contornos que parecem colunas na imagem pré-processada."""
    if len(processed_image.shape) == 3:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    all_contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtra contornos por proporção (colunas são altas e finas)
    filtered_contours = []
    for c in all_contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Garante que 'h' é o maior lado (altura) para cálculo da proporção
        if w > h:
            w, h = h, w

        # Filtro de proporção: W/H >= 1/4 (coluna razoavelmente fina/alta)
        if w / h >= 1/4:
            filtered_contours.append(c)

    # Seleciona os dois maiores contornos em área
    filtered_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return filtered_contours[:2]


def find_column_rectangles(processed_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encontra os 4 vértices aproximados (retângulo de área mínima) das duas principais colunas.
    """
    # Se a imagem for em escala de cinza/binária, precisamos de BGR para o desenho,
    # mas o detector de contornos só precisa de 1 canal.
    contours = find_main_column_contours(processed_image)
    
    column_rect_points: list[np.ndarray] = []

    for contour in contours:
        # Encontra o retângulo de área mínima
        min_area_rect = cv2.minAreaRect(contour)
        
        # Obtém os 4 vértices (box points)
        box_points = cv2.boxPoints(min_area_rect)
        
        # Ordena os pontos (TL, TR, BR, BL)
        ordered_points = order_points(box_points.astype(np.float32))

        column_rect_points.append(ordered_points)
    
    # Garante que a coluna da esquerda esteja na primeira posição
    col_left, col_right = column_rect_points[0], column_rect_points[1]
    if col_left[0][0] > col_right[0][0]:
        col_left, col_right = col_right, col_left
    
    return col_left, col_right


def find_columns(image: np.ndarray, preprocess_method: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Função principal. Pré-processa a imagem, detecta colunas e corrige a perspectiva.
    Retorna a imagem da coluna esquerda e da coluna direita.
    """
    if preprocess_method == 1:
        preprocessed_image = preprocess_image(image)
    else:
        preprocessed_image = preprocess_image_v2(image)
    
    # Encontra os vértices das colunas
    col_left_points, col_right_points = find_column_rectangles(preprocessed_image)

    # Aplica a correção de perspectiva para ambas as colunas
    col_left_img = apply_perspective_warp(image, col_left_points)
    col_right_img = apply_perspective_warp(image, col_right_points)
    
    return col_left_img, col_right_img

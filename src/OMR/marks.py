import cv2
import numpy as np
from typing import Optional

from .omr_config import (
    DEBUG, ALTERNATIVAS, QUESTIONS_PER_COLUMN, COLUMN_WIDTH,
    Y_START_QUESTION_1, Y_SPACING_PER_QUESTION, CONTOUR_AREA_MIN_MARK
)

MarkData = tuple[int, str, int, int, int, int]  # QuestionNumber, Value, X, Y, W, H


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Aplica o pré-processamento de imagem para realçar marcações em uma única linha."""
    # Turn image to gray
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Aplica fechamento morfológico para preencher pequenas lacunas na marcação
    kernel_close = np.ones((3, 3), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel_close, iterations=5)

    # Erosão para refinar bordas
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_ERODE, kernel_erode, iterations=3)

    # Ajusta o contraste para melhor separação de fundo/marcação
    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=1.2, beta=30)

    # Threshold
    thresh_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 197, 10)
  
    # Erosão e Fechamento finais para limpeza e destaque
    kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_final = cv2.morphologyEx(thresh_image, cv2.MORPH_ERODE, kernel_rect_3, iterations=3)

    kernel_rect_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    output_image = cv2.morphologyEx(eroded_final, cv2.MORPH_CLOSE, kernel_rect_2, iterations=1)
    
    return output_image


def _find_mark_contours(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Encontra e filtra contornos que representam marcações (bolhas preenchidas)."""
    # Converte para escala de cinza se não estiver
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra contornos por área. Assumimos um range de área adequado para a bolha
    filtered_contours = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 700 < cv2.contourArea(c) < 4800 and w > 20 and h > 27:
            filtered_contours.append(c)
    
    return tuple(filtered_contours)


def _map_x_to_alternative(x_coord: int, line_width: int) -> Optional[str]:
    """Mapeia a coordenada X (relativa à linha) para a alternativa (A-E) mais próxima."""
    num_alternatives = len(ALTERNATIVAS)
    
    # Normaliza X para o range [0, num_alternatives-1]
    alternative_index = int(x_coord / line_width * num_alternatives)
    
    if 0 <= alternative_index < num_alternatives:
        return ALTERNATIVAS[alternative_index]
    return None


def _crop_question(image: np.ndarray, question_index: int) -> np.ndarray:
    """Recorta a subimagem correspondente à linha de uma única questão."""
    y_start = int(Y_START_QUESTION_1 + question_index * Y_SPACING_PER_QUESTION)
    y_end = int(y_start + Y_SPACING_PER_QUESTION)
    
    height = image.shape[0]
    y_end = min(y_end, height)
    
    line_image = image[y_start:y_end, 0:COLUMN_WIDTH]
    
    return line_image


def find_marks(column_image: np.ndarray) -> list[MarkData]:
    """
    Detecta todas as marcações em uma imagem de coluna, retornando o valor e 
    as coordenadas absolutas de cada marca individual.
    """
    marks: list[MarkData] = []
    
    for question_index in range(QUESTIONS_PER_COLUMN):
        question_number = question_index + 1
        
        # Uso dos aliases internos para manter o estilo da sua sugestão
        question_image = _crop_question(column_image, question_index)
        
        if question_image.size == 0:
            continue
            
        preprocessed = _preprocess_image(question_image)
        contours = _find_mark_contours(preprocessed)

        y_offset = int(Y_START_QUESTION_1 + question_index * Y_SPACING_PER_QUESTION)
        line_width = question_image.shape[1]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour) # Coordenadas relativas à linha            
            center_x = int(x + w / 2)
            
            value = _map_x_to_alternative(center_x, line_width)
            
            if value is not None:
                x_original = x
                y_original = y + y_offset
                
                # question_number é o número da questão DENTRO da coluna (1 a 15)
                marks.append((question_number, value, x_original, y_original, w, h))
                
    return marks


def _group_marks_by_question(all_marks: list[MarkData], column_offset: int) -> dict[int, tuple[Optional[str], list[MarkData]]]:
    """Agrupa as marcações e determina a resposta final para cada questão."""
    grouped_marks: dict[int, tuple[Optional[str], list[MarkData]]] = {}
    
    # Agrupamento e determinação da resposta (string)
    marks_temp: dict[int, list[MarkData]] = {}
    alternatives_temp: dict[int, set[str]] = {}

    for q_num_col, value, x, y, w, h in all_marks:
        q_num_abs = q_num_col + column_offset
        
        marks_temp.setdefault(q_num_abs, []).append((q_num_col, value, x, y, w, h))
        alternatives_temp.setdefault(q_num_abs, set()).add(value)

    # Finaliza a resposta (string) e estrutura o retorno
    for q_num_abs, alt_set in alternatives_temp.items():
        if len(alt_set) == 0:
            result_answer = None
        elif len(alt_set) == 1:
            result_answer = list(alt_set)[0]
        else:
            result_answer = ','.join(sorted(alt_set))
            
        grouped_marks[q_num_abs] = (result_answer, marks_temp[q_num_abs])

    return grouped_marks


def get_answers(column_images: tuple[np.ndarray, ...]) -> dict[int, Optional[str]]:
    """
    Processa todas as colunas, detecta as marcações e retorna o dicionário de respostas (JSON).
    """
    all_answers: dict[int, Optional[str]] = {}
    
    for i, col_img in enumerate(column_images):
        # Detecção (MarkData)
        marks_data = find_marks(col_img)
        
        # Agrupamento e Resposta
        column_offset = i * QUESTIONS_PER_COLUMN
        grouped_marks = _group_marks_by_question(marks_data, column_offset)
        
        # Extração final do JSON
        for q_num_abs, (result_answer, _) in grouped_marks.items():
            all_answers[q_num_abs] = result_answer
            
    return all_answers


def paint_marks(column_images: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """
    Processa as colunas e retorna as imagens pintadas com as marcações detectadas.
    Formato: quadrado verde com texto "{q_number}: {answer}".
    """
    painted_columns: list[np.ndarray] = []

    for i, original_image in enumerate(column_images):
        # 1. Detecção (MarkData)
        marks_data = find_marks(original_image)
        
        # 2. Agrupamento e Resposta
        column_offset = i * QUESTIONS_PER_COLUMN
        grouped_marks = _group_marks_by_question(marks_data, column_offset)
        
        output_img = original_image.copy()

        # 3. Desenho
        for q_num_abs, (result_answer, marks_in_q) in grouped_marks.items():
            if result_answer is None:
                continue

            for _, _, x_original, y_original, w, h in marks_in_q:
                
                # Desenha o retângulo (Verde: 0, 255, 0)
                cv2.rectangle(output_img, (x_original, y_original), (x_original + w, y_original + h), (0, 255, 0), -1)

                # Desenha o texto do resultado (Preto)
                text_label = f'{q_num_abs}: {result_answer}'
                text_position = (x_original + 8, y_original + int(h/2) + 8)
                cv2.putText(
                    img=output_img,
                    text=text_label,
                    org=text_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 0), 
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        
        # Desenhar linhas guias (Para visualização)
        for q_idx in range(QUESTIONS_PER_COLUMN):
            y_start = int(Y_START_QUESTION_1 + q_idx * Y_SPACING_PER_QUESTION)
            y_end = int(y_start + Y_SPACING_PER_QUESTION)
            cv2.rectangle(output_img, (0, y_start), (COLUMN_WIDTH, y_end), (255, 0, 0), 2)

        painted_columns.append(output_img)
        
    return tuple(painted_columns)

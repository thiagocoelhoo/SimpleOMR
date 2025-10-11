import cv2
import numpy as np
from typing import List, Tuple
import math

# --- CONSTANTES GLOBAIS DE CONFIGURAÇÃO ---
# Parâmetros para detecção da FOLHA (recortar_gabarito)
CONTOUR_AREA_THRESHOLD_SHEET = 1000
KERNEL_SIZE_SHEET = (20, 20)
CLOSING_ITERATIONS_SHEET = 3

# Parâmetros para detecção das MARCAÇÕES (contornar_respostas)
CONTOUR_AREA_MIN_MARK = 50
ADAPTIVE_THRESH_BLOCK_SIZE = 141
ADAPTIVE_THRESH_C = 10
KERNEL_SIZE_MARK = (3, 3)
CONTRAST_ALPHA = 1.5
BRIGHTNESS_BETA = 25

# Coordenadas X das colunas de alternativas na imagem JÁ RECORTADA
# Assumimos que o gabarito é alinhado e padronizado após o 'crop'.
# O valor 240 é o ponto de corte entre as colunas esquerda e direita.
X_SPLIT_COLUMN = 240
COORDS_ALTERNATIVAS_X = [
    53,     # A (coluna 1)
    81,     # B (coluna 1)
    115,    # C (coluna 1)
    150,    # D (coluna 1)
    182,    # E (coluna 1)
    334,    # A (coluna 2)
    370,    # B (coluna 2)
    403,    # C (coluna 2)
    428,    # D (coluna 2)
    472,    # E (coluna 2)
]
ALTERNATIVAS = tuple('ABCDE')

# CONSTANTES PARA O MAPPING Y -> QUESTÃO
Y_START_QUESTION_1 = 5
Y_SPACING_PER_QUESTION = 30
QUESTIONS_PER_COLUMN = 15
Y_MARGIN_ERROR = 15

def rotate_image(img: np.ndarray, center: Tuple[float, float], angle: float, scale: float = 1.0) -> np.ndarray:
    """
    Gira uma imagem em torno de um ponto central.
    """
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image


class OmrProcessor:
    """
    Processa uma imagem de folha de respostas (gabarito) para detectar,
    corrigir a perspectiva e extrair as respostas preenchidas.
    """

    def __init__(self, alternative_coords_x: List[int] = COORDS_ALTERNATIVAS_X):
        """Inicializa o processador com as coordenadas X esperadas."""
        self.alternative_coords_x = alternative_coords_x
    
    def preprocess_img(self, img: np.ndarray):
        # # 1. Pré-processamento e Ajuste de Contraste
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # Ajusta brilho e contraste para realçar as marcações escuras
        # gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)

        # # 2. Limiarização Adaptativa
        # # Usada para isolar marcações (preenchimentos) escuras em papel mais claro.
        # thresh = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
        #     ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
        # )
        
        # # 3. Operação Morfológica: Erosão
        # # Ajuda a remover ruídos e a garantir que os contornos sejam apenas das marcações.
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_MARK)
        # closed = cv2.erode(thresh, kernel)
        # ------------------------------------------------------------------------

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Thresholding
        # Usamos BINARY_INV para que as áreas escuras (o objeto/folha) fiquem brancas (255)
        # para detecção de contornos.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 10
        )
        
        # 2. Operação Morfológica: Fechamento (Closing)
        # Preenche pequenos buracos e junta áreas próximas da folha para melhorar a detecção de contorno.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_SHEET)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERATIONS_SHEET)


        return closed

    def _get_sheet_countor(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Thresholding
        # Usamos BINARY_INV para que as áreas escuras (o objeto/folha) fiquem brancas (255)
        # para detecção de contornos.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 10
        )
        
        # 2. Operação Morfológica: Fechamento (Closing)
        # Preenche pequenos buracos e junta áreas próximas da folha para melhorar a detecção de contorno.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_SHEET)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERATIONS_SHEET)

        # 3. Detecção de Contornos (Find Contours)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0

        # 4. Seleção do Melhor Contorno
        # OBS: O melhor contorno é a maior área,
        #      pois o gabarito é o maior elemento da página.
        for c in contours:
            area = cv2.contourArea(c)
            # Filtra por uma área mínima para ignorar ruído
            if area > CONTOUR_AREA_THRESHOLD_SHEET and area > max_area:
                max_area = area
                best_contour = c

        if best_contour is None:
            raise Exception("A área principal do gabarito não foi detectada. Tente ajustar os parâmetros (kernel, area_minima).")

        return best_contour
    
    def align_and_crop_sheet(self, img: np.ndarray) -> np.ndarray:
        """
        Localiza a folha de respostas na imagem, corrige a rotação e recorta.

        :param img: Imagem de entrada da folha de respostas.
        :return: A imagem da folha de respostas alinhada e recortada.
        :raises Exception: Se o gabarito principal não for detectado.
        """
        output = img.copy()

        best_contour = self._get_sheet_countor(img)
        
        # 5. Correção de Rotação e Recorte
        # Obtém o retângulo com a menor área que envolve o contorno
        rect = cv2.minAreaRect(best_contour)
        (center, (width, height), angle) = rect
        
        # Normaliza o ângulo para garantir que a folha fique na vertical
        # Se a folha estiver em pé (portrait), o ângulo precisa ser corrigido
        # if width < height:
        #     angle = 90 + angle
        if angle > 60 or angle < -60:
            angle = (angle % 90) - 90
        
        # Obtém o retângulo delimitador (x, y, largura, altura) para o recorte final
        x, y, w, h = cv2.boundingRect(best_contour)

        # Corrige o ângulo da imagem
        # rotated_image = output
        rotated_image = rotate_image(output, center, angle - 1.5, 1.0)
        best_contour = self._get_sheet_countor(rotated_image)
        x, y, w, h = cv2.boundingRect(best_contour)

        # Recorta a área de interesse
        TOP_PADDING = int(0.0945179584120983 * h) # Padding para ignorar o cabeçalho do gabarito
        cropped_image = rotated_image[y + TOP_PADDING : y + h, x : x + w]
        cropped_image = cv2.resize(cropped_image, dsize=(513, 479))
        return cropped_image

    def _get_question_number(self, y_coord: float, x_coord: int) -> int:
        """
        Mapeia uma coordenada Y para um número de questão (1 a 30)
        e ajusta para a coluna (esquerda: 1-15, direita: 16-30).
        """
        # 1. Determina a posição relativa na coluna (0 para Q1, 1 para Q2, etc.)
        # Fórmula: (Y - Y_início) / Delta Y
        relative_position = (y_coord - Y_START_QUESTION_1) / Y_SPACING_PER_QUESTION

        # Arredonda para o número da questão mais próximo (de 1 a 15)
        question_number = int(round(relative_position))
        
        # O número da questão deve estar no intervalo esperado (1 a 15)
        question_number = max(1, min(QUESTIONS_PER_COLUMN, question_number))
        
        # 2. Ajusta o número da questão para a coluna correta (1-30)
        if x_coord > X_SPLIT_COLUMN:
            question_number = question_number + QUESTIONS_PER_COLUMN # Coluna Direita: Questões 16 a 30
        
        return question_number

    def _map_x_to_alternative(self, x):
        # Mapeia a coordenada X da marcação para a alternativa mais próxima
        nearest_distance = float('inf')
        alternative_value = None

        for i, c_x in enumerate(self.alternative_coords_x):
            # Compara a posição X do contorno (x) com a coordenada X esperada (c_x)
            if abs(c_x - x) < nearest_distance:
                nearest_distance = abs(c_x - x)
                # O módulo 5 garante o ciclo A, B, C, D, E
                alternative_value = ALTERNATIVAS[i % 5]
        
        return alternative_value
    
    def detect_and_map_answers(self, img: np.ndarray) -> Tuple[np.ndarray, dict[int, str | None]]:
        """
        Detecta as marcações na imagem do gabarito recortado e as mapeia para alternativas.

        :param img: Imagem do gabarito já alinhado e recortado.
        :return: Tupla contendo (imagem com contornos desenhados, lista ordenada de respostas).
        """
        marks = dict()
        output_img = img.copy()

        # kernel = np.ones((4, 4),np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 1. Pré-processamento e Ajuste de Contraste
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ajusta brilho e contraste para realçar as marcações escuras
        gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)

        # 2. Limiarização Adaptativa
        # Usada para isolar marcações (preenchimentos) escuras em papel mais claro.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
            ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
        )
        
        # 3. Operação Morfológica: Erosão
        # Ajuda a remover ruídos e a garantir que os contornos sejam apenas das marcações.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_MARK)
        closed = cv2.erode(thresh, kernel)

        # 4. Detecção dos contornos das marcações (Find Contours)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filtragem e mapeamento baseado na coordenada da marcação
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtra por área mínima (exclui pequenos ruídos)
            if area > CONTOUR_AREA_MIN_MARK: 
                x, y, w, h = cv2.boundingRect(contour)
                alternative_value = self._map_x_to_alternative(x)
                
                question_number = self._get_question_number(y + h / 2, x) # Usamos o centro Y da marcação
                if question_number > 0 and alternative_value is not None:
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), -1)
                    
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    textPosition =(x + 4, y+20)

                    # Draw the text on the image
                    cv2.putText(
                        img=output_img,
                        text=f'{question_number}-{alternative_value}',
                        org=textPosition,
                        fontFace=fontFace,
                        fontScale=0.3,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
                    
                    #  Armazena a resposta detectada
                    if alternative_value is not None:
                        if marks.get(question_number):
                            marks[question_number] += ',' + alternative_value
                        else:
                            marks[question_number]  = alternative_value
    
        # 6. Ordenação Final
        return output_img, marks

    def process_image(self, img: np.ndarray):
        img_gabarito = self.align_and_crop_sheet(img)
        output_img, responses_json = self.detect_and_map_answers(img_gabarito)

        for i in range(1, 31):
            if responses_json.get(i) is None:
                responses_json[i] = None
        
        return output_img, responses_json

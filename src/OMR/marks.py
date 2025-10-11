import cv2
import os
import numpy as np
from typing import Dict, Any, Tuple

from omr_config import *

def show(img: np.ndarray, force: bool = False, title: str = 'img', wait: bool = True):
    """Exibe a imagem para depuração."""
    if DEBUG or force:
        resized = cv2.resize(img, dsize=None, fx=0.4, fy=0.4)
        cv2.imshow(title, resized)
        if wait:
            cv2.waitKey(0)


def process_image(img: np.ndarray) -> np.ndarray:
    """Aplica o pré-processamento para destacar as marcações."""
    # Step 1: gray
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # show(gray)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=5)
    # show(gray)

    # Step 2: adjust contrars
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    # show(gray)

    # Step 4: thresh
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 141, 10)
    # show(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel, iterations=3)
    # show(gray)

    # Step 6: close and erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    output = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    # show(output)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    output = cv2.morphologyEx(output, cv2.MORPH_ERODE, kernel, iterations=3)
    # show(output)

    return output


def find_marks_contours(img: np.ndarray) -> Tuple[np.ndarray, ...]:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = []
    for contour in contours:
        if 700 < cv2.contourArea(contour) < 4800:
            output.append(contour)
    
    return tuple(output)

    
def map_x_to_alternative(x: int, img_width: int) -> str | None:
    """Mapeia a coordenada X da marcação para a alternativa mais próxima (A-E) 
    dentro da largura da linha (subimagem)."""
    # Se a largura da imagem for 400, x / 400 * 5 calcula o índice da alternativa
    # A largura da subimagem de uma linha é WIDTH_COLUMN (400)
    alternative_index = int(x / img_width * len(ALTERNATIVAS))
    
    if 0 <= alternative_index < len(ALTERNATIVAS):
        return ALTERNATIVAS[alternative_index]
    return None


def extract_question_line(image: np.ndarray, question_index: int) -> np.ndarray:
    """
    Recorta a imagem para isolar a linha de uma única questão.
    O índice da questão (0 a 14) é usado para calcular as coordenadas Y.
    """
    # Coordenadas Y (topo e base) da linha da questão
    y_start = int(Y_START_QUESTION_1 + question_index * Y_SPACING_PER_QUESTION)
    y_end = int(y_start + Y_SPACING_PER_QUESTION)
    
    # Garantir que as coordenadas estejam dentro dos limites da imagem
    height, width, *_ = image.shape
    y_end = min(y_end, height)
    
    # Recorta a subimagem
    # Assumimos que a coluna tem largura total (0 a WIDTH_COLUMN)
    line_image = image[y_start:y_end, 0:WIDTH_COLUMN]
    
    return line_image


def extract_answers_per_line(original_image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str | None]]:
    """
    Processa a imagem dividindo-a em linhas de questões, 
    processando cada linha separadamente e mapeando as respostas.
    """
    marks: Dict[int, str | None] = {}
    output_img = original_image.copy()
    
    # Itera sobre cada questão (índice 0 a 14)
    for question_index in range(QUESTIONS_PER_COLUMN):
        # O número real da questão (1 a 15)
        question_number = question_index + 1
        
        # 1. Extrair a linha da questão
        line_image = extract_question_line(original_image, question_index)
        
        if line_image.size == 0:
            print(f"Aviso: Linha da questão {question_number} vazia.")
            marks[question_number] = None
            continue

        # 2. Pré-processar a linha
        preprocessed_line = process_image(line_image)
        # show(preprocessed_line, title=f"Q{question_number} Processada")
        
        # 3. Encontrar contornos na linha
        contours = find_marks_contours(preprocessed_line)
        
        # Coordenadas Y reais na imagem original
        y_offset = int(Y_START_QUESTION_1 + question_index * Y_SPACING_PER_QUESTION)
        
        detected_alternatives: list[str] = []
        
        # 4. Mapear contornos para alternativas
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtro de área mais estrito é vital aqui, pois processamos apenas uma pequena área
            # Ajuste CONTOUR_AREA_MIN_MARK se necessário, mas 50 é um bom ponto de partida.
            if area > CONTOUR_AREA_MIN_MARK:
                x_line, y_line, w, h = cv2.boundingRect(contour) # Coordenadas relativas à linha
                
                # Centro da marcação
                center_x_line = int(x_line + w / 2)
                
                # Mapeia X para a alternativa (usando a largura da linha)
                alternative_value = map_x_to_alternative(center_x_line, line_image.shape[1])
                
                if alternative_value is not None:
                    # Coordenadas absolutas para desenhar na imagem original
                    x_original = x_line
                    y_original = y_line + y_offset
                    center_x_original = center_x_line
                    center_y_original = int(y_line + h / 2) + y_offset
                    
                    # Desenha o retângulo na imagem de saída
                    cv2.rectangle(output_img, (x_original, y_original), (x_original + w, y_original + h), (0, 255, 0), -1)
                    
                    # Desenha o texto do resultado
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    textPosition = (x_original + 8, y_original + int(h/2) + 8)
                    cv2.putText(
                        img=output_img,
                        text=f'{question_number}-{alternative_value}',
                        org=textPosition,
                        fontFace=fontFace,
                        fontScale=0.8,
                        color=(0, 0, 0),
                        thickness=2, # Reduzido para melhor visualização
                        lineType=cv2.LINE_AA
                    )
                    
                    if alternative_value not in detected_alternatives:
                         detected_alternatives.append(alternative_value)

        # 5. Armazena a resposta final para a questão
        if len(detected_alternatives) == 0:
            marks[question_number] = None
        elif len(detected_alternatives) == 1:
            marks[question_number] = detected_alternatives[0]
        else:
            # Marcação dupla/múltipla
            marks[question_number] = ','.join(sorted(detected_alternatives))

    # Desenha as linhas guias na imagem de saída (opcional, para visualização)
    for i in range(QUESTIONS_PER_COLUMN):
        cv2.rectangle(output_img, (0, int(i * Y_SPACING_PER_QUESTION)), (WIDTH_COLUMN, int((i+1) * Y_SPACING_PER_QUESTION)), (255, 0, 0), 2)
        
    return output_img, marks


def extract_answers(image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str | None]]:
    """Função de alto nível para extrair as respostas usando o novo método linha-a-linha."""
    return extract_answers_per_line(image)


def main():
    try:
        from columns import find_columns, perspective_warp
    except ImportError:
        print("Erro: Módulos 'find_columns' e 'perspective_warp' não encontrados. Certifique-se de que estão disponíveis.")
        return
    


    # O resto da lógica principal permanece o mesmo
    # imagens = ['/home/thiago/Downloads/WhatsApp Image 2025-10-10 at 19.27.29(3).jpeg']
    imagens = [
        '/home/thiago/Downloads/WhatsApp Image 2025-10-10 at 20.07.09(1).jpeg',
        '/home/thiago/Downloads/WhatsApp Image 2025-10-10 at 19.27.29(3).jpeg',
        '/home/thiago/Downloads/WhatsApp Image 2025-10-10 at 19.27.29(2).jpeg',
        '/home/thiago/Downloads/WhatsApp Image 2025-10-10 at 19.27.30(1).jpeg',
    ]
    
    # for root, _, files in os.walk('/home/thiago/Downloads/'):
    #     for file in files:
    #         if file.endswith('jpeg'):
    #             imagens.append(os.path.join(root, file))
    
    for img_path in imagens:
        img = cv2.imread(img_path)
        show(img, force=True)

        if img is None:
            print("Erro ao carregar a imagem. Verifique o caminho.")
            return
        
        left_col_rect, right_col_rect = find_columns(img)

        left_img = perspective_warp(img, left_col_rect)
        right_img = perspective_warp(img, right_col_rect)
        
        # Processa as duas colunas
        imagens = [left_img, right_img]

        if not imagens:
            print("Nenhuma imagem de coluna processada.")
            return

        all_answers = {}

        for i, image in enumerate(imagens):
            # A nova função extract_answers usa o método linha-a-linha
            output_image, answers = extract_answers(image)

            # Ajusta os números das questões para a segunda coluna (16-30)
            if i == 1:
                adjusted_answers = {q + QUESTIONS_PER_COLUMN: a for q, a in answers.items()}
                all_answers.update(adjusted_answers)
            else:
                all_answers.update(answers)

            show(output_image, force=True)

        # Print do resultado final
        print("\nRespostas Detectadas (Todas as Questões):")
        print(img_path)
        for q, a in sorted(all_answers.items()):
            # Exibe 'Não Marcada' para None
            display_a = a if a is not None else 'Não Marcada' 
            print(f"Q{q:02}: {display_a}")
        print("---------------------------------")


if __name__ == '__main__':
    main()
    
# Limpa as janelas abertas
if __name__ == '__main__':
    cv2.destroyAllWindows()
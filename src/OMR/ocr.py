import easyocr

reader = easyocr.Reader(['pt'], gpu=False) # Defina gpu=True se tiver uma GPU compatível para acelerar

def find_matricula(image):
    """
    Localiza uma matrícula (string numérica)
    com base em critérios de tamanho e confiança usando EasyOCR.
    """

    # Reconhecimento de Texto com EasyOCR
    results = reader.readtext(image)
    print('results:', results)

    for (bbox, text, conf) in results:    
        # Coordenadas do bounding box
        x_min = int(bbox[0][0])
        y_min = int(bbox[0][1])
        x_max = int(bbox[2][0])
        y_max = int(bbox[2][1])

        # Largura e Altura do texto reconhecido
        w = x_max - x_min
        h = y_max - y_min

        if conf > 0.7 and h > 20 and text.isdecimal():
            return text
            
    return None

import cv2
import os


# Diretório onde as imagens capturadas serão armazenadas
output_directory = "imagens_capturadas"

# Certifique-se de que o diretório de saída exista, ou crie-o se necessário
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Carregue o classificador pré-treinado para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialize a câmera
cap = cv2.VideoCapture(0)

while True:
    # Capture um quadro da câmera
    ret, frame = cap.read()

    # Converta o quadro para escala de cinza (melhor para detecção facial)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte faces no quadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhe retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Capture a imagem da face frontal
        face_image = frame[y:y + h, x:x + w]
        
        # Salve a imagem no diretório de saída
        image_filename = os.path.join(output_directory, f"face_{len(os.listdir(output_directory)) + 1}.jpg")
        cv2.imwrite(image_filename, face_image)
        
    # Exiba o quadro resultante
    cv2.imshow('Detecção Facial', frame)

    # Pare o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()

import cv2
import face_recognition
import numpy as np 

try:
    image_eronaldo = face_recognition.load_image_file("Pessoas/eronaldo.jpg")
    encoding_eronaldo = face_recognition.face_encodings(image_eronaldo)[0]

    image_maria = face_recognition.load_image_file("Pessoas/maria.jpg")
    encoding_maria = face_recognition.face_encodings(image_maria)[0]

    known_face_encodings = [encoding_eronaldo, encoding_maria]
    known_face_names = ["Eronaldo", "Maria"]
    
except (FileNotFoundError, IndexError) as e:
    print(f"Erro ao carregar rostos conhecidos: {e}")
    print("Verifique se os arquivos de imagem existem e se há um rosto neles.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

while True:
    # Captura um único quadro de vídeo
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o quadro. Encerrando...")
        break

    # Converte a imagem de BGR (padrão do OpenCV) para RGB (padrão do face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontra todas as localizações de rostos no quadro atual
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        # Calcula os encodings para os rostos encontrados
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Itera sobre cada rosto encontrado no quadro
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compara o rosto encontrado com os rostos conhecidos
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            name = "Desconhecido" # Nome padrão se não houver correspondência

            # Encontra a melhor correspondência
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Escreve o nome do rosto identificado
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Detecção Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

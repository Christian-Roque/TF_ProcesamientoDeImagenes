import cv2
import face_recognition

def detect_faces():
    # Abre la cámara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usa cv2.CAP_DSHOW como backend

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame (stream end?). Exiting ...")
            break

        # Convierte la imagen de BGR a RGB (face_recognition usa RGB)
        rgb_frame = frame[:, :, ::-1]

        # Encuentra todas las ubicaciones de los rostros en el frame
        face_locations = face_recognition.face_locations(rgb_frame)

        # Dibuja rectángulos alrededor de cada rostro detectado
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Muestra el frame con los rostros detectados
        cv2.imshow('Face Detection', frame)

        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la captura de video y cierra todas las ventanas OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()

import cv2

def test_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # '0' selecciona la cámara predeterminada

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame (stream end?). Exiting ...")
            break

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()

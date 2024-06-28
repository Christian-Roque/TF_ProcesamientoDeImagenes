import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('stayalert_model.h5')  

def detect_faces():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame (stream end?). Exiting ...")
            break

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
       
            face_image = frame[top:bottom, left:right]
            resized_face = cv2.resize(face_image, (227, 227))
            normalized_face = resized_face / 255.0
            input_data = np.expand_dims(normalized_face, axis=0)
            prediction = model.predict(input_data)
          
            if prediction < 0.5:
                label = "No Drowsy"
                color = (0, 255, 0)  
            else:
                label = "Drowsy"
                color = (0, 0, 255)  

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


        cv2.imshow('Drowsiness Detection', frame)

     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()


import cv2

# Cargar el clasificador preentrenado de OpenCV para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video de la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer un frame de la cámara web
    ret, frame = cap.read()

    # Convertir el frame a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar el frame con los rostros resaltados
    cv2.imshow('Face Tracking', frame)

    # Si se presiona la tecla 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara web y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
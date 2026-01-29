"""
Asistencia con reconocimiento facial dependiendo de la hora que se realiza,
no es soportado por versiones de python mayores a python 3.11
cv2 y face_recognition no son compatibles con versiones mayores.
"""
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

#cargar caras conocidas (colocar fotos en dataset, carpeta con dataset debe tener nombre de la persona)

### compilar el archivo, Q para salir de la camara

ruta_dataset = "dataset"
nombres_conocidos = []
rostros_conocidos = []

for persona in os.listdir(ruta_dataset):
    ruta_persona = os.path.join(ruta_dataset, persona)

    if not os.path.isdir(ruta_persona):
        continue

    imagen_path = os.path.join(ruta_persona, os.listdir(ruta_persona)[0])

    imagen = face_recognition.load_image_file(imagen_path)
    encoding = face_recognition.face_encodings(imagen)

    if encoding:
        rostros_conocidos.append(encoding[0])
        nombres_conocidos.append(persona)
        print(f"[OK] Rostro cargado: {persona}")
    else:
        print(f"[ERROR] No se detectó rostro en {persona}")

#registrar asistencia(mirar csv de asistencia para ver registros)

def registrar_asistencia(nombre):
    with open("asistencia.csv", "r+") as archivo:
        lineas = archivo.readlines()
        nombres_registrados = [linea.split(",")[0] for linea in lineas]

        if nombre not in nombres_registrados:
            ahora = datetime.now()
            fecha = ahora.strftime("%Y-%m-%d")
            hora = ahora.strftime("%H:%M:%S")
            archivo.write(f"{nombre},{fecha},{hora}\n")
            print(f"[ASISTENCIA] {nombre} registrada")

#crear archivo si no existe
if not os.path.exists("asistencia.csv"):
    with open("asistencia.csv", "w") as f:
        f.write("Nombre,Fecha,Hora\n")


#abrir camara

camara = cv2.VideoCapture(0)

while True:
    ret, frame = camara.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ubicaciones = face_recognition.face_locations(frame_rgb)
    encodings = face_recognition.face_encodings(frame_rgb, ubicaciones)

    for encoding, ubicacion in zip(encodings, ubicaciones):
        comparaciones = face_recognition.compare_faces(rostros_conocidos, encoding)
        nombre = "Desconocido"

        if True in comparaciones:
            indice = comparaciones.index(True)
            nombre = nombres_conocidos[indice]
            registrar_asistencia(nombre)

        top, right, bottom, left = ubicacion
        cv2.rectangle(frame,
                      (left, top),
                      (right, bottom),
                      (0, 255, 0),
                      2)
        cv2.putText(frame,
                    nombre,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Control de Asistencia", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

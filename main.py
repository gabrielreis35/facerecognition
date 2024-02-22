import cv2
import face_recognition as fr
import os
import time
import numpy as np
from PIL import Image
from psycopg2 import OperationalError

cap = cv2.VideoCapture(0)

preset_folder_path = './preset'

compare_time = time.time()
verify_time = time.time()

pre_set_images = []

def connect_db():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="postgres",
            host="teste",
            port="5432",
            database="teste"
        )
        return connection
    except OperationalError as e:
        print(e)
        return None
    
def disconnect_db(connection):
    if connection:
        connection.close()
        return True
    
    return False

def consult_db(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    except OperationalError as e:
        print(e)
        return None
    finally:
        cursor.close()

def load_preset_images():
    global pre_set_images
    pre_set_images = os.listdir(preset_folder_path)
    
def verify_face(compare_frame):
    for image in pre_set_images:
        path_image = os.path.join(preset_folder_path, image)

        preset_image = fr.load_image_file(path_image)
        preset_image = cv2.cvtColor(preset_image, cv2.COLOR_BGR2RGB)

        encode_preset = fr.face_encodings(preset_image)
        if encode_preset:
            encode_preset = encode_preset[0]

            encode_image = fr.face_encodings(compare_frame)
            if encode_image:
                encode_image = encode_image[0]
                compare = fr.compare_faces([encode_image], encode_preset)

                print(compare)

load_preset_images()
# connect = connect_db()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    now = time.time()
    if now - compare_time > 30:
        load_preset_images()
        compare_time = now

    frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame_array)
    temp_file_path = os.path.join(os.getcwd(), "temp_frame.jpg")
    img.save(temp_file_path)

    compare_frame = fr.load_image_file(temp_file_path)
    os.remove(temp_file_path)
    compare_frame = cv2.cvtColor(compare_frame, cv2.COLOR_BGR2RGB)

    if len(fr.face_locations(compare_frame)) > 0:
        face_loc = fr.face_locations(compare_frame)[0]
        cv2.rectangle(compare_frame, (face_loc[3], face_loc[0], face_loc[1], face_loc[2]), (0, 255, 0), 2)
        
    verify_face(compare_frame)

    cv2.imshow("teste", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

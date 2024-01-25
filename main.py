import tempfile
import cv2
import face_recognition as fr
import os
import time
import threading
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)

preSetImages = []
compareImages = []
presetFolder = os.listdir('./preset')
compareFolder = os.listdir('./compare')

compareTime = time.time()

def VerificaImagemPasta():
    presetFolder = os.listdir('./preset')
    for item in presetFolder:
        preSetImages.append(item)
    print(len(preSetImages))

# for image in compareFolder:
#     compareImages.append(image)

threading.Thread(target=VerificaImagemPasta).start()

while (cap.isOpened()):
    ret, frame = cap.read()
    
    if not ret:
        break
    
    now = time.time()
    if now - compareTime > 300:
        threading.Thread(target=VerificaImagemPasta).start()
    
    frame_array = np.array(frame)
    
    img = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
    temp_file_path = os.path.join(tempfile.gettempdir(), "temp_frame.jpg")
    img.save(temp_file_path)
    
    compareFrame = fr.load_image_file(temp_file_path)
    os.remove(temp_file_path)
    compareFrame = cv2.cvtColor(compareFrame, cv2.COLOR_BGR2RGB)
    
    if len(fr.face_locations(compareFrame)) > 0:
        faceLoc = fr.face_locations(compareFrame)[0]
        cv2.rectangle(compareFrame, (faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
    
    for image in preSetImages:
        pathImage = "./preset/" + str(image)
        
        presetImage = fr.load_image_file(pathImage)
        presetImage = cv2.cvtColor(presetImage, cv2.COLOR_BGR2RGB)
        
        encodePreset = fr.face_encodings(presetImage)[0]
        
        encodeImage = fr.face_encodings(compareFrame)
        if encodeImage:
            encodeImage = encodeImage[0]
            compare = fr.compare_faces([encodeImage], encodePreset)
        
        print(compare)
        
    cv2.imshow("teste", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



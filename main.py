import cv2
import face_recognition as fr
import os
import time

cap = cv2.VideoCapture(0)

preSetImages = []
compareImages = []
presetFolder = os.listdir('./preset')
compareFolder = os.listdir('./compare')

compareTime = time.time()

def VerificaImagemPasta():
    presetFolder = os.listdir('./preset')
    # if os.path.exists(presetFolder) == False:
    #     return False
    for item in presetFolder:
        preSetImages.append(item)

for image in compareFolder:
    compareImages.append(image)

VerificaImagemPasta()

while (cap.isOpened()):
    ret, frame = cap.read()
    
    if not ret:
        break
    
    now = time.time()
    if now - compareTime > 300:
        VerificaImagemPasta()
        
    for image in compareImages:
        pathImage = "./compare/" + str(image)
        
        compareImage = fr.load_image_file(pathImage)
        compareImage = cv2.cvtColor(compareImage, cv2.COLOR_BGR2RGB)
        
        presetImage = fr.load_image_file(frame)
        presetImage = cv2.cvtColor(presetImage, cv2.COLOR_BGR2RGB)
    
        faceLoc = fr.face_locations(compareImage)[0]
    
        cv2.rectangle(compareImage, (faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
        
        encodePreset = fr.face_encodings(presetImage)[0]
        encodeImage = fr.face_encodings(compareImage)[0]
        
        compare = fr.compare_faces([encodeImage], encodePreset)
        
        print(compare)
    
        # print(encodePreset)
        cv2.imshow("teste", compareImage)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()



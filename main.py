import cv2
import face_recognition as fr
import os

preSetImages = []
preSetFolder = os.listdir('./preset')

for item in preSetFolder:
    preSetImages.append(item)

for item in preSetImages:
    path = "./preset/" + str(item)
    imageTrained = fr.load_image_file(path)
    imageTrained = cv2.cvtColor(imageTrained, cv2.COLOR_BGR2RGB)
    
    faceLoc = fr.face_locations(imageTrained)[0]
    
    cv2.rectangle(imageTrained, (faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
    encodeImage = fr.face_encodings(imageTrained)[0]
    
    # compare = fr.compare_faces([encodeImage], encodeCompare)
    
    print(encodeImage)
    cv2.imshow("teste", imageTrained)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
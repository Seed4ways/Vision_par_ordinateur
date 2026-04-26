# Imports Bibliothèque
import onnxruntime
import os
# import dlib
import cv2
import numpy as np 
from ultralytics import YOLO

# Models :
    # Import class:

from ressources.models.yolov8face.detector_v2 import YOLOv8_face 
    # Import model
    # choix des paramètres:
    
    # + le conf_tresh est élevé plus le risque de ne pas detecté un visage est élevé
    # et - le risque de détecter comme un visage autre chose est faible 
    # ici dans la mesure ou par la suite nous allons détecter le visage, puis les émotions 
    # je prends la décision de prendre un seuil relativement élevé car les visages incertains
    # ne seront surement pas viable pour la detection des émotions

    # en fonction de la luminosité la confiance du model diminue grandement 
    # à adapter en fonction de la lumiosité de l'endroit

    # + le iou_tresh est élevé - on detecte et élimine de doublons avec NMS
    # ici dans la mesure ou le seuil de confiance que j'ai choisi est plutôt élevé
    # j'ai moins besoin d'éliminer de doublon via NMS et je peux donc prendre un 
    # seuil relativement élevé pour IOU (intersection over union)
    # Ce qui permet de moins utilisé NMS et donc de limité le temps d'éxecution

model = YOLOv8_face(path="ressources/models/yolov8face/yolov8n-face.onnx", 
                    conf_thres=0.1, # 
                    iou_thres=0.6,  # Intersection over union 
                    show_keypoint=False)

def affichage_YOLO_ROI(img, det_bboxes, det_conf):
    
    # det_bboxes est une liste du tuple (x, y, w, h)
    # ou x et y sont l'abscisse et l'ordonnée du point superieur gauche
    # et w,h la largeur et la hauteur de la bounding box

    for index, box in enumerate(det_bboxes): # j'utilise enumerate pour obtenir opt
            
            x, y, w, h = box
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

            # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
            # Org est le coin bas gauche du rectangle
            cv2.putText(img, (f'confiance du model: {det_conf[index]}'),(x-40, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,0),2 )

            # amélioration changer le placement du texte en fonction de la taille de la bounding box


    #return # les numpy array sont passées par référence, cv2.rectangle s'applique donc sur l'image en mémoire et 
           # il n'y a pas de nécessité à retourner l'image


# Cam setting 
cam = 0 # le channel caméra
cap = cv2.VideoCapture(cam,cv2.CAP_V4L2)

if not cap.isOpened():
    print("could not reach camera")
    exit()


# Boucle de traitement 
while True:

    # Lecture de l'image dans le flux video
    ret, img = cap.read()

    # Verification, s'il n'y a pas d'image exit()
    if not ret:
        print('img couldnt be captured')
        exit()

    # Detection des visage 
    # https://github.com/chumpblocckami/blur_face/blob/main/detector.py
    # return det_bboxes, det_conf, det_classid, landmarks
    # Bboxes = bounding boxes = ROIS
    # det Conf = confiance en la detection 
    # det_classid ici toujours 0 car nous ne detectons que la classe face
    det_bboxes, det_conf, det_classid, landmarks = model.detect(img) 

    # Affichage Roi visage
    img_roi = img.copy()
    affichage_YOLO_ROI(img_roi,det_bboxes,det_conf) 



    

    cv2.imshow("img",img_roi)


    key = cv2.waitKey(1) & 0xFF # 0xFF == mask
    # explanation :
    # waitKey gives back a 32 bits answer 
    # 24 of which are not the info we need 
    # ( the key code ex: 27 )
    # answer format 0000000 00000000 00000000 10110100
    # & = and operator 
    # 0xFF = 00000000 00000000 00000000 11111111
    # answer & 0xFF = the last 8 bits of the answer 
    if key == ord('q'): # ord('q') gives the ascii number 
        break


cap.release()
cv2.destroyAllWindows()

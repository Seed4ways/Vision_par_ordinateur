
# Imports Bibliothèques
import os 
import cv2
import numpy as np 
from ultralytics import YOLO
import onnxruntime


from ressources.models.yolov8face.detector_v2 import YOLOv8_face 

yolov8 = YOLOv8_face(path="ressources/models/yolov8face/yolov8n-face.onnx", 
                    conf_thres=0.1, # 
                    iou_thres=0.6,  # Intersection over union 
                    show_keypoint=False)


# Load Model OpenFace
openFace = cv2.dnn.readNetFromTorch("Codes/openface.nn4.small2.v1.t7")

def calcul_desc(img,roi):
    # normaliserez (i.e. vous transformerez l’image 
    # pour quelle soit au format attendu par OpenFace) 
    # ce ROI par la fonction dnn.blobFromImage.
    #  
    # L’image sera en couleur, au format 96x96, 
    # le facteur d’échelle 1/255, la moyenne (0,0,0) et swapRB=True

    x,y,w,h = roi # on prends les coordonnées de la 1ère ROI, detecter dans l'image
    img_roi = img[y:y+h, x:x+w] # on extrait cette region de l'image

    # on la reformatte pour le réseau
    blob = cv2.dnn.blobFromImage(
        img_roi,
        scalefactor=1/255.0,
        size=(96, 96),
        mean=(0, 0, 0),
        swapRB=True
    )
    openFace.setInput(blob)
    desc = openFace.forward() # descripteur
    return desc.flatten()

def distance(desc1,desc2):
    d = desc1 - desc2
    distance = np.dot(d, d)
    return distance

def database(db_path):

    names = []
    descripteurs = []

    if not os.path.exists(db_path):
        print(f"⚠️  Dossier '{db_path}' introuvable.")
        return names, descripteurs

    # Pour chaque nom de dossier dans la liste des dossiers
    for person_name in os.listdir(db_path): 
        person_path = os.path.join(db_path, person_name)


        if not os.path.isdir(person_path):
            continue
        
        # Pour chaque fichier image dans la liste d'un dossier de la bd
        for img_file in os.listdir(person_path):
            # on créer son paths
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Détection du visage dans l'image de la base
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conversion en niveau de gris
            gray = cv2.equalizeHist(gray) # normalisation 
            det_bboxes, det_conf, det_classid, landmarks = yolov8.detect(img)  # detection 

            if len(det_bboxes) == 0:
                print(f"Aucun visage détecté dans {img_path}")
                continue

            # On prend le premier visage détecté
            x, y, w, h = det_bboxes[0]
            face_roi = img[y:y+h, x:x+w]  # ROI en couleur BGR

            desc = calcul_desc(face_roi)
            descripteurs.append(desc)
            names.append(person_name)
            print(f"✅  Descripteur calculé pour {person_name} ({img_file})")

    return names, descripteurs



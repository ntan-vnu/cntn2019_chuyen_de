import cv2
import os
import glob
import numpy as np

def load_uid():
    folders = glob.glob('data/*/')
    folders.sort()
    folders = [it.split('/')[1] for it in folders]
    dict_uid = {}
    for index, it in enumerate(folders):
        dict_uid[index] = it
    return dict_uid
    pass

if __name__ == "__main__":
    cap = cv2.VideoCapture('harry-potter.mp4')
    face_cascade = cv2.CascadeClassifier('cvdata/haarcascade_frontalface_default.xml')
    
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read('model.yml')
    dict_uid = load_uid()
    print(dict_uid)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    while(count < num_frames):
        ret, img = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in faces:
            if w < 64 or h < 64:
                continue
            crop_face = img[y:y+h, x:x+w, :]
            crop_face = cv2.resize(crop_face, (128, 128))
            crop_face = np.array(crop_face, dtype=np.float64) / 255
            crop_face = np.reshape(crop_face, (-1))
            pre, score = recognizer.predict(crop_face)
            if score >= 0.75:
                uid = dict_uid[pre]
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                img = cv2.putText(img, uid, (x, y),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0,255,0), 1)

        count += 1

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pass
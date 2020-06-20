import glob
import cv2

def crop_face(filename):
    img = cv2.imread(filename) # BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        'cvdata/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if (len(faces)) > 0:
        max_face = faces[0]
        for (x, y, w, h) in faces:
            if w*h > max_face[2]*max_face[3]:
                max_face = (x, y, w, h)
        crop_img = img[y:y+h, x:x+w, :]
        crop_face = cv2.resize(crop_face, (128, 128))
        cv2.imwrite(filename, crop_img)
    else:
        print(filename)
    pass

if __name__ == "__main__":
    files = glob.glob('data/*/*')
    files.sort()
    for it in files[:5]:
        print(it)
    
    for it in files:
        crop_face(it)

    pass
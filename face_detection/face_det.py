import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('webcame', frame)
        cv2.waitKey(0)

    
    cap.release()
    pass
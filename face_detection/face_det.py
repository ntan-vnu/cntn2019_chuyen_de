import cv2

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier('cvdata/haarcascade_frontalface_default.xml')

	while(True):
		ret, img = cap.read()

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)

		for x, y, w, h in faces:
			img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()
	pass
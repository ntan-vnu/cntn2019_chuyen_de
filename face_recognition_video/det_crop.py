import cv2

if __name__ == "__main__":
	cap = cv2.VideoCapture('harry-potter.mp4')
	face_cascade = cv2.CascadeClassifier('cvdata/haarcascade_frontalface_default.xml')
	
	count = 0
	while(count < 300):
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
			cv2.imwrite('data/%05d.jpg'%(count), crop_face)
			count += 1

			# img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			

		# cv2.imshow('frame',img)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
			# break
	
	cap.release()
	cv2.destroyAllWindows()
	pass
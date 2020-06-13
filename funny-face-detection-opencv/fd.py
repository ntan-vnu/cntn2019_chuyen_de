import numpy as np
import cv2

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img

if __name__ == '__main__':
	face_cascade = cv2.CascadeClassifier('cvdata/haarcascade_frontalface_default.xml')
	video_capture = cv2.VideoCapture(0)

	devface = cv2.imread('img/face.png', -1)
	_h, _w, _ = devface.shape
	count = 0
	while True:
		ret, img = video_capture.read()
		count += 1
		if not ret or count%15 == 0:
			continue

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)

		try:
			for (x,y,w,h) in faces:
				tmp_devface = cv2.resize(devface, (0, 0), fx=float(w)/_w * 1.25, fy=float(h)/_h * 1.25)
				px = x + w / 2 - tmp_devface.shape[1] / 2
				py = y + h / 2 - tmp_devface.shape[0] / 2
				img = overlay_transparent(img, tmp_devface, px, py)
				roi_gray = gray[y:y+h, x:x+w]
				roi_color = img[y:y+h, x:x+w]
		except:
			continue
		cv2.imshow('live',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
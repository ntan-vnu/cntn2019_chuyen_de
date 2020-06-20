# pip install opencv-contrib-python

import numpy as np
import cv2
import sys
import glob

def load_dict_uid():
	dict_uid = {}
	fds = [it.replace('data/', '') for it in  glob.glob('data/*')]
	fds.sort()
	for index, it in enumerate(fds):
		dict_uid[it] = index
	
	return dict_uid

def load_data_samples(meta_file, dict_uid):
	images = []
	labels = []
	with open(meta_file) as f:
		for line in f:
			filename, uid = line.strip().split('\t')
			img = cv2.imread(filename) / 255
			img = np.reshape(img, (-1))
			images.append(img)
			labels.append(dict_uid[uid])

	return images, labels

def train(meta_file):
	# create uid dict
	dict_uid = load_dict_uid()
	print(dict_uid)

	# load data samples
	images, labels = load_data_samples(meta_file, dict_uid)

	# create model --> train
	recognizer = cv2.face.EigenFaceRecognizer_create()
	recognizer.train(images, np.array(labels))

	# save model
	recognizer.save('model.yml')
	
def test(file_model, meta_file):
	recognizer = cv2.face.EigenFaceRecognizer_create()
	recognizer.read(file_model)

	dict_uid = load_dict_uid()
	images, labels = load_data_samples(meta_file, dict_uid)
	count = 0
	for i in range(len(images)):
		pre, score = recognizer.predict(images[i])
		count += 1 if pre==labels[i] else 0   # <>? <True> : <False>

	return count / len(labels)

	pass

if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == 'train':
		train('train.txt')
	print('train acc:', test('model.yml', 'train.txt'))
	print('test acc:', test('model.yml', 'test.txt'))
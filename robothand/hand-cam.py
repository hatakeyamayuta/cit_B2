import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from  keras.preprocessing.image import img_to_array
from  keras.preprocessing.image import load_img
import keras
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import paramiko

def hand_write(n):

    with paramiko.SSHClient() as ssh:

        #AutoAddPolicyで勝手にhostname & key を登録してもらう
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #ssh接続する
        ssh.connect('10.232.169.107',port=22,username='ubuntu',password='raspberry')
        #shell芸でファイルに値書き込み
        stdin,stdout,stderr=ssh.exec_command("cd /run/shm; echo %s > angles"%(n[0]) )
        stdin,stdout,stderr=ssh.exec_command("cd /run/shm; echo %s > ev_on_off"%(n[1]) )

def theta():

	k=[[ 0 for i in range(2)] for j in range(10)]

	k[0]=['0,60,30,-90,8', 0]
	k[1]=['45,84,39,-34,0', 0]
	k[2]=['45,120,10,-40,30', 0]
	k[3]=['45,120,10,-40,30', 1]
	k[4]=['45,84,39,-34,0', 1]
	k[5]=['-45,84,39,-34,0', 1]
	k[6]=['-45,120,10,-40,30', 1]
	k[7]=['-45,120,10,-40,30', 0]
	k[8]=['-45,84,39,-34,0', 0]
	k[9]=['0,60,30,-90,8', 0]

	for i in range(10):
		n=k[i]
		hand_write(n)
		print(n[0])
		print(n[1])
		if input() == 1:
			continue
	
	return	

def predict():

	model=Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10,activation='softmax'))

	return model

def transform_by4(img, points):
	""" 4点を指定してトリミングする。 """

	points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
	top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
	bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
	points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。

	width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
	height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))

	dst = np.array([
			np.array([0, 0]),
			np.array([width-1, 0]),
			np.array([width-1, height-1]),
			np.array([0, height-1]),
			], np.float32)

	# 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
	trans = cv2.getPerspectiveTransform(points, dst)


	return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。


if __name__ == '__main__':
	cam = cv2.VideoCapture(0)

	while cv2.waitKey(10) != 27:
		orig = cam.read()[1]

		lines = orig.copy()

		# 輪郭を抽出する
		canny = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
		canny = cv2.GaussianBlur(canny, (5, 5), 0)
		canny = cv2.Canny(canny, 50, 100)
		cv2.imshow('canny', canny)

		# 抽出した輪郭に近似する直線（？）を探す。
		AAA, cnts, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts.sort(key=cv2.contourArea, reverse=True)  # 面積が大きい順に並べ替える。

		warp = None
		
		for i, c in enumerate(cnts):
			arclen = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02*arclen, True)

			level = 1 - float(i)/len(cnts)  # 面積順に色を付けたかったのでこんなことをしている。
			if len(approx) == 4:

				cv2.drawContours(lines, [approx], -1, (0, 0, 255*level), 2)
				if warp is None:
					warp = approx.copy()  # 一番面積の大きな四角形をwarpに保存。
			else:
				cv2.drawContours(lines, [approx], -1, (0, 255*level, 0), 2)

			for pos in approx:
				cv2.circle(lines, tuple(pos[0]), 4, (255*level, 0, 0))

		cv2.imshow('edge', lines)

		if warp is not None:
			
			# warpが存在した場合、そこだけくり抜いたものを作る。
			warped = transform_by4(orig, warp[:,0,:])
			cv2.imshow('warp', warped)

			cv2.imwrite("trimming.png", warped)

			if cv2.waitKey(10) == ord('s'):

				image_path ='trimming.png'
				img = load_img(image_path, grayscale = True, target_size=(28,28))
				x=img_to_array(img)
				x /= 255
				x = np.expand_dims(x,axis=0)
				print(x.shape)

				model = predict()
				model.summary()
				model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
				model.load_weights('mnist.h5')

				features = model.predict_classes(x, batch_size=32)
				print(features)
				print(np.argmax(features))

				if features == 3:
					theta()
				else:
					continue

	cam.release()
	cv2.destroyAllWindows()

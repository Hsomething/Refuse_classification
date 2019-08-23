import time
import threadpool
import os
import cv2

name = os.listdir('after/')

def create_read_img(filename) :
	img = cv2.imread('after/'+filename)
	out_90 = rotate(img,90)
	out_180 = rotate(img,180)
	out_270 = rotate(img,270)

	cv2.imwrite("after/"+filename[:-4]+'_90.jpg', out_90)
	cv2.imwrite("after/"+filename[:-4]+'_180.jpg', out_180)
	cv2.imwrite("after/"+filename[:-4]+'_270.jpg', out_270)
	print(filename)

def rotate(image, du):
	'''
		旋转
	'''
	rows, cols = image.shape[:2]
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2),du, 1)
	dst = cv2.warpAffine(image, M, (cols, rows))
	return dst


start = time.time()
pool = threadpool.ThreadPool(5)
requests = threadpool.makeRequests(create_read_img,name)
[pool.putRequest(req) for req in requests]
pool.wait()
print ('%d second'% (time.time()-start))
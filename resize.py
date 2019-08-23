from PIL import Image
import os
dir_list = os.listdir('干垃圾/')
#print(dir_list)
width,height = 224,224
for row in dir_list:
	img = Image.open('干垃圾/'+row)
	new_img=img.resize((width,height),Image.BILINEAR)   
	new_img.save('after/'+row)
	print(row)
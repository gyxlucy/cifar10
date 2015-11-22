import Image
import sys


arr = []

for i in range(0,32*32):
	for j in ['r','g','b']:
		arr.append('"px' + j + str(i) + '"')
arr.append('"class"')
print ",".join(arr)

for x in range(1, 300000+1):

	#im = Image.open('train/' + str(x) + '.png').convert('LA')
	im = Image.open('test/' + str(x) + '.png')

	#im.show()
	
	arr = []
	for i in range(0, 32):
		for j in range(0, 32):
			tp = im.getpixel((i,j))
			arr.append( str(tp[0]) )
			arr.append( str(tp[1]) )
			arr.append( str(tp[2]) )
			
	print ",".join(arr) + "," + labels[x]

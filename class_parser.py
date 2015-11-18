import csv

original_labels = open('trainLabels.csv', 'rb')
original_pixels = open('train.csv', 'rb')
new_labels = open('trainLabels_modified.csv', 'wb')
new_pixels = open('train_modified.csv', 'wb')


labelreader = csv.reader(original_labels)
pixelsreader = csv.reader(original_pixels)

labelwriter = csv.writer(new_labels)
pixelswriter = csv.writer(new_pixels)

labelwriter.writerow(labelreader.next())
pixelswriter.writerow(pixelsreader.next())

classes = ['frog', 'ship', 'airplane', 'deer']
counter = 1
for row in labelreader:
	pixel_row = pixelsreader.next()
	if row[1] in classes:
		row[0] = counter
		labelwriter.writerow(row)
		pixelswriter.writerow(pixel_row)
		counter += 1
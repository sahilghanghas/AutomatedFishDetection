import fnmatch
import os
import csv

images = []
annotations = []

for file in os.listdir('/media/auv/DATA/Data/Corrected'):
	fileBasename = os.path.splitext(file)[0]
	images.append(fileBasename)


for file in os.listdir('/media/auv/DATA/Data/Corrected/Annotations'):
 	fileBasename = os.path.splitext(file)[0]
 	annotations.append(fileBasename)
print len(annotations)

with open('test.csv', 'wb') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',',
		quotechar="|", quoting=csv.QUOTE_MINIMAL)
	for annotation in annotations:
		if annotation in images:
			filewriter.writerow([annotation])
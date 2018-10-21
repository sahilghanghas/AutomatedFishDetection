import fnmatch
import os
import csv
import sys

images = []
annotations = []
annotated = []
unannotated = []

fn1 = sys.argv[1]
fn2 = sys.argv[2]

for file in os.listdir(fn1):
	fileBasename = os.path.splitext(file)[0]
	images.append(fileBasename)


for file in os.listdir(fn2):
 	fileBasename = os.path.splitext(file)[0]
 	annotations.append(fileBasename)
print len(annotations)

for image in images:
		if image in annotations:
			annotated.append(image)
		else:
			unannotated.append(image)

print "Number of annotated images"
print len(annotated)

print "Number of unannotated images"
print len(unannotated)

with open('annotated.csv', 'wb') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',',
		quotechar="|", quoting=csv.QUOTE_MINIMAL)
	for annotation in annotated:
		filewriter.writerow([annotation])

with open('unannotated.csv', 'wb') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',',
		quotechar="|", quoting=csv.QUOTE_MINIMAL)
	for annotation in unannotated:
		filewriter.writerow([annotation])

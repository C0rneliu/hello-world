import sys
import cv2
import os
import mysql.connector
import time
import shutil
from numpy import *
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from matplotlib import *
#style.use("ggplot")
from sklearn import svm
from sklearn.externals import joblib
from time_log_debug import *

clf = svm.SVC(kernel='rbf', gamma=0.007, class_weight={0: 1.0, 1: 0.1})

def TrainingFunction(TrainingPath):
    TrainingData = []
    time_log ("Read training images...")
    listing = sorted(os.listdir(TrainingPath))
    for index, file in enumerate(listing):
        if index == 10000:
			break
		im = cv2.imread(TrainingPath + str(index + 1) + ".png")
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            TrainingData.append((im))
        else:
            time_log("Could not open or find the image...")
            exit(-1)

    time_log ("Extract features...")
    winSize = (24, 24)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    HOGdescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    v_descriptorsValues = []
    Traininglabels = np.ones((len(TrainingData), ), dtype=np.int)
    for i in range(0, len(TrainingData)):
        # HOG extraction
        descriptorsValues = HOGdescriptor.compute(TrainingData[i], (0, 0), (0, 0))
        v_descriptorsValues.append(descriptorsValues)
        # training labels
        if (i < 5000):
            Traininglabels[i] = 1
        else:
            Traininglabels[i] = 0

  # prepare training set
    nsamples, nx, ny = np.shape(v_descriptorsValues)
    TrainingSet = np.reshape(v_descriptorsValues, (nsamples, nx*ny))

    # classifier training
    time_log ("Start training...")
    clf.fit(TrainingSet[:10000], Traininglabels[:10000])
    joblib.dump(clf, 'TrainedRejector.pkl')
    time_log ("Finished training...")
    return clf

def TestingFunction(mypath):
    TestingData = []
    for img in glob.glob(mypath + "*.png"):
        im = cv2.imread(img)
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im.resize((48, 48))
            TestingData.append((im))
        else:
            time_log ("Could not open or find the image...")
            exit(-1)

    winSize = (16, 16)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    HOGdescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    v_descriptorsValues = []

    for i in range(0, len(TestingData)):
        # HOG extraction
        descriptorsValues = HOGdescriptor.compute(TestingData[i], (0, 0), (0, 0))
        v_descriptorsValues.append(descriptorsValues)

    # prepare training set
    nsamples, nx, ny = np.shape(v_descriptorsValues)
    TestingSet = np.reshape(v_descriptorsValues, (nsamples, nx * ny))
    
	clf = joblib.load('TrainedRejector.pkl')
	
	label = clf.predict(TestingSet)

    classification = {}

    for i, fileName in enumerate(os.listdir(mypath)):
        if i == len(TestingSet):
            break
     	classification[fileName] = label[i]

    return classification


TrainingPath = str(sys.argv[1])
TrainingFunction(TrainingPath)


#Establish database connection
cnx = mysql.connector.connect(user='root', password='MovonUcec5', host='127.0.0.1', database='atv_db_impax')

# Interogare baza de date, copiere imagine in director temporar, scanare, actualizare baza de date, stergere director temporar

while (1):
	mypath = r'/home/cornel/Rej_test/Analyzer/'
	exp_path = r'/home/impax/impressions/'
#	shutil.rmtree(mypath)

	if not os.path.isdir(mypath):
			os.makedirs(mypath)

	cnx = mysql.connector.connect(user='root', password='MovonUcec5', host='127.0.0.1', database='atv_db_impax')
	cursor = cnx.cursor()
	time_log ("Connection established...")
	
	tests = 20
	query_worker_update = ("UPDATE event_scanner_4 SET worker = 'scn4' WHERE worker IS NULL AND status = 'new' LIMIT " + str(tests))
	query_select = ("SELECT * FROM event_scanner_4 WHERE status = 'new' ORDER BY added ASC LIMIT " + str(tests))
	time_log ("Querries done...")

#	cursor.execute(query_worker_update)
#	cnx.commit()
	time_log ("Query worker update, done...")

	cursor.execute(query_select)
	rows = cursor.fetchall()
#	print(rows)
	time_log("Fetch, done..")

	for event_id in rows:
		file_name = str(event_id[1]) + ".png"
		time_log(file_name)
	#	If file doesn't exists on drive
		
		copier = 0
		if (os.path.isfile(exp_path + file_name)):
			shutil.copy2(exp_path + file_name, mypath + file_name)
			copier += 1
		else:
			query_is_Nofile = ("UPDATE event_scanner_4 SET updated = NOW(), status = 'error' WHERE event_id = " + str(event_id[1]))
			time_log (query_is_Nofile)
			cursor.execute(query_is_Nofile)
			cnx.commit()
	time_log('Exit for loop...')

	classifications = {}
	if (copier > 0):
		time_log('Classifing...')
		classifications = TestingFunction(mypath)
		event_id = classifications.keys()
		scanner4_face = classifications.values()
		time_log("Finished...")

	for i in range (0, len(classifications)):
		query_final_update = ("UPDATE event_scanner_4 SET scanner4_face = " + str(1 - scanner4_face[i]) + ", updated = NOW(), status = 'finished' WHERE event_id = " + "'" + str(event_id[i]) + "'")
		time_log (query_final_update)
		cursor.execute(query_final_update)
		cnx.commit()

	shutil.rmtree(mypath)

	cnx.close()

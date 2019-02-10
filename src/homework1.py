# -*- coding: utf-8 -*-
"""
AI & ML Homework - prof. Caputo, year 2018
Homework 1 
@student: Antonio Tavera 
@id: 243869

Created on Sat Nov 10 19:51:17 2018
"""

import os, sys 
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
import random
from PIL import Image

#defining variables
data_path = '../PACS_homework/'
label_name = []		#corresponding name-label index
test_percentage = 0.2	#test percentage for splitting
X_training = []		#training set
y_training = []		#training labels
trainingdata_mean = []	#mean array
trainingdata_std = []	#std array



###############   Load the dataset  ################
# load all the dataset from the data_path directory
# visit directory recursively and save dataset into
# X_train. Create also labels for the dataset
# Returns X_train and y_train
#
#####################################################
def load_dataset():
	X_train = []
	y_train = [] 
	for root, dirs, files in os.walk(data_path):
		dirs.sort()
		path = root.split('/')
		if(path[2]!=''):
			label_name.append(path[2])
			files.sort()
			for file in files: 
				img_path = root + str('/') + file 
				img_data = np.asarray(Image.open(img_path))
				img_data = img_data.ravel()
				X_train.append(img_data)
				y_train.append(label_name.index(path[2]))
	#convert from list to array	
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
		
	return X_train, y_train


###############   Split the dataset  ################
# receive X_train and y_train, suffle data
# take the test percentage data from X_train for 
# the test set, order data and returns X_train,
# y_train, X_test, y_test
# 
#####################################################
def split_dataset(X_train, y_train):
	#shuffle data 
	indices = np.arange(0, len(X_train))
	random.shuffle(indices)
	X_train = X_train[indices]
	y_train = y_train[indices]

	#take last test percentage% to test set 
	test_size = int(np.round(test_percentage * len(X_train)))
	X_test = X_train[:test_size]
	y_test = y_train[:test_size]
	X_train = X_train[test_size:]
	y_train = y_train[test_size:]
	
	#sort data according to labels in ascending order
	indeces = np.argsort(y_train)
	X_train = X_train[indeces]
	y_train = y_train[indeces]
	indeces = np.argsort(y_test)
	X_test = X_test[indeces]
	y_test = y_test[indeces]
	
	return  X_train, y_train, X_test, y_test


###########  Create and train the model #############
# receive X_train and y_train, create the Gaussian 
# Naive Bayes classifier model and train that model 
# with the received dataset
# Returns the trained model 
# 
#####################################################
def train(X_train, y_train): 
	gnb_model = GaussianNB()
	gnb_model.fit(X_train, y_train)
	
	return gnb_model


################  Test the model  ###################
# Receives the trained model, the X_test dataset and 
# its labels. Make a prediction on the model and 
# compute and print the accuracy. Returns the accuracy
# 
#####################################################
def test_model(model, X_test, y_test):
	#make a prediction on the test set
	test_predictions = model.predict(X_test)
	test_predictions = np.round(test_predictions)
	
	#report the accuracy of that prediction 
	accuracy = accuracy_score(y_test, test_predictions)
	print("Test set accuracy: %.4f"  %(accuracy)) 
	return accuracy 


################  Normalize Data  ###################
# Receives the data to be normalize
# Subtract the mean and divide by the standard 
# deviation. Update the trainingdata_mean and _std 
# array to be capable of doing the unnormalization. 
# Returns the normalized training set 
# 
#####################################################
def normalize_data(X_train):
	train_set = []
	i = 0
	for image in X_train:
		data = np.array(image).copy()
		trainingdata_mean.append(np.mean(image))
		trainingdata_std.append(np.std(image))
		data = data - trainingdata_mean[i]
		data = data / trainingdata_std[i]
		train_set.append(data)
		i += 1
		
	train_set = np.asarray(train_set)	
	return train_set


################  Unnormalize Data  #################
# Receives the data to be unnormalize
# Multiply by the saved std and add the mean 
# Returns the unnormalized training set 
# 
#####################################################
def unnormalize_data(X_train):
	train_set = []
	i = 0
	for image in X_train: 
		data = np.array(image).copy()
		data = data * trainingdata_std[i]
		data = data + trainingdata_mean[i]
		train_set.append(np.uint8(data))
		i += 1
	
	train_set = np.asarray(train_set)
	return train_set 


############### Compute first N PC  #################
# Receives as input the training set and the number 
# of the first N components to compute
# Normalize data, compute the N components, project 
# data on those components, transform the dataset and 
# unnormalize data. Returns the original dataset 
# reprojected and the trasnformed data
# 
#####################################################
def compute_first_PC(X_train, num_components):
	#standardize data
	train_set = normalize_data(X_train)
	#set pca components
	pca = PCA(n_components=num_components)
	#extract principal components
	#not compute the fit_transform because I already centralized data 
	pca.fit(train_set)
	#transform
	data_transformed = np.dot(train_set, pca.components_.T)
	#reproject data with pc
	data_original = np.dot(data_transformed, pca.components_) 
	#unstandardize data
	data_original = unnormalize_data(data_original)
	
	return data_original, data_transformed


######## Compute last or from/to PC  ################
# Receives as input the training set and the from and
# to principal component to be computed; if the to_pc 
# variable is set to None compute the last component 
# using as number of components the from_or_last_pc
# variable. Normalize data, compute the chosen 
# components, transform data on those components, 
# reproject the dataset and unnormalize data. 
# Returns the original dataset reprojected and the
#  trasnformed data
# 
#####################################################
def compute_PC(X_train, from_or_last_pc, to_pc=None):
	#standardize data
	train_set = normalize_data(X_train)
	#set pca components to nothing, it compute all the principal components
	pca = PCA()
	#extract all principal components
	#not compute the fit_transform because I already centralized data 
	pca.fit(train_set)
	
	if(to_pc != None):
		#select from i to j principal components
		from_pc = from_or_last_pc - 1
		computed_pc = pca.components_[from_pc:to_pc, :]
	else:
		#select last n principal components
		to_pc = len(pca.components_)
		from_pc = to_pc - from_or_last_pc
		computed_pc = pca.components_[from_pc:to_pc, :]
	
	#transform
	data_transformed = np.dot(train_set, computed_pc.T)
	#reproject data with pc, inverse transform
	data_original = np.dot(data_transformed, computed_pc) 
	#unstandardize data
	data_original = unnormalize_data(data_original)
	
	return data_original, data_transformed
	

############### Display an image ####################
# Receives as input the image to be displayed and 
# its label(optionally). Reshape the image to the 
# original form and display it 
# 
#####################################################
def display_img(data, label=None): 
	#reshape image to original form
	plt.figure()
	data = np.reshape(data, (227, 227, 3))
	plt.imshow(data)
	if(label!=None):
		print(" The image is a: " + str(label_name[label]))
	#plt.savefig('figure_%s.jpg' %(label_name[label]), dip=500)
	plt.show()
		
	return 
	

############### Display N images  ###################
# Receives as input the images to be displayed and 
# their label(optionally). Reshape the images to their
# original form and display it in a single figure
# 
#####################################################
def show_images(images, cols=1, titles=None, label=None):
	assert((titles is None)or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		image = np.reshape(image, (227, 227, 3))
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
		a.set_title(title, fontsize=50, ha='center')
		
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.savefig('figure_%s.jpg' %(label), dip=500)
	plt.show()
	
	return



##################    MAIN     ######################
#####################################################
####################  menu  #########################
# 1. Load the whole dataset and creates labels
# 2. For each class, compute the requested PCs and 
#    visualize a random chosen image
# 3. For all the dataset, compute the requested PCs
#    and visualize a random chosen image
# 4. Visualize transformed data after PCA using scatter
# 5. Classify data with GNB classifier
# 6. Exit the process
#####################################################	
while(1):
	print('\n')
	print("***** MENU *****")
	menu_choice = str(input('1. Load Dataset\n'\
		         + '2. PC Visualization with separated class\n'\
			 + '3. PC Visualization with all dataset\n'\
			 + '4. Scatter plot transformed data\n'\
			 + '5. GNB Classification\n'\
			 + '6. Exit\n'\
			 + 'Input: '))


	if menu_choice == '1':
		#Load the whole dataset and generate labels
		X_training, y_training = load_dataset()
	
	
	elif menu_choice == '2':
		#PC Visualization with separated class
		if len(X_training) == 0:
			X_training, y_training = load_dataset()
			
		for name in label_name:
			class_index = label_name.index(name)
			#select all the class data from the train set and test set 
			X_training_sep = X_training[y_training == class_index].copy()
			y_training_sep = y_training[y_training == class_index].copy()
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca2, transf = compute_first_PC(X_training_sep, 2)
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca6, transf = compute_first_PC(X_training_sep, 6)
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca60, transf = compute_first_PC(X_training_sep, 60)
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pcalast6, transf = compute_PC(X_training_sep, from_or_last_pc=6)
			
			#show one random image reproject with different pc
			index = random.randint(0, len(y_training_sep))
			images = []
			titles = []
			images.append(X_training_sep[index])
			titles.append(str("Original"))
			images.append(X_train_pca2[index])
			titles.append(str("2 PCA"))
			images.append(X_train_pca6[index])
			titles.append(str("6 PCA"))
			images.append(X_train_pca60[index])
			titles.append(str("60 PCA"))
			images.append(X_train_pcalast6[index])
			titles.append(str("Last 6 PCA"))
			show_images(images, 1, titles, name)
	
	
	elif menu_choice == '3'	:
		#PC Visualization with all dataset
		if len(X_training) == 0:
			X_training, y_training = load_dataset()
		#compute PCA and reproject image using PC
		trainingdata_mean = []
		trainingdata_std = []
		X_train_pca2, X_pca2_transf = compute_first_PC(X_training, 2)
		X_train_pca6, transf = compute_first_PC(X_training, 6)
		X_train_pca60, transf = compute_first_PC(X_training, 60)
		X_train_pcalast6, transf = compute_PC(X_training, from_or_last_pc=6)
		
		#print a random chosen image from all the training set
		index = random.randint(0, len(y_training))
		images = []
		titles = []
		images.append(X_training[index])
		titles.append(str("Original"))
		images.append(X_train_pca2[index])
		titles.append(str("2 PCA"))
		images.append(X_train_pca6[index])
		titles.append(str("6 PCA"))
		images.append(X_train_pca60[index])
		titles.append(str("60 PCA"))
		images.append(X_train_pcalast6[index])
		titles.append(str("Last 6 PCA"))
		show_images(images, 1, titles, label_name[y_training[index]])
		
		
	elif menu_choice == '4':
		#Scatter plot PC transformed data
		
		if len(X_training) == 0:
			X_training, y_training = load_dataset()
		
		#set scatter plot variable
		color_dict = ['yellow','red','magenta','blue']
		
		#f, axarr = plt.subplots(3, sharex=True, sharey=False)
		plt.figure(1, figsize=(30, 10))
		#scatter plot data first 2 PC
		i=0
		ax1 = plt.subplot(131)
		ax1.set_title('Scatter 2PC', ha='center')
		for name in label_name:
			class_index = label_name.index(name)
			#select all the class data from the train set and test set 
			X_training_sep = X_training[y_training == class_index].copy()
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca, X_pca_transf = compute_first_PC(X_training_sep, 2)
			ax1.scatter(X_pca_transf[:,0], X_pca_transf[:,1], c=color_dict[i])
			i+=1
		
		#scatter plot data 3 to 4 PC
		i=0
		ax2 = plt.subplot(132)
		ax2.set_title('Scatter 3&4PC', ha='center')
		for name in label_name:
			class_index = label_name.index(name)
			#select all the class data from the train set and test set 
			X_training_sep = X_training[y_training == class_index].copy()
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca, X_pca_transf = compute_PC(X_training_sep, 3, 4) 
			ax2.scatter(X_pca_transf[:,0], X_pca_transf[:,1], c=color_dict[i])
			i+=1
		
		#scatter plot from 10 to 11 PC
		i=0
		ax3 = plt.subplot(133)
		ax3.set_title('Scatter 10&11PC', ha='center')
		for name in label_name:
			class_index = label_name.index(name)
			#select all the class data from the train set and test set 
			X_training_sep = X_training[y_training == class_index].copy()
			trainingdata_mean = []
			trainingdata_std = []
			X_train_pca, X_pca_transf = compute_PC(X_training_sep, 10, 11) 
			ax3.scatter(X_pca_transf[:,0], X_pca_transf[:,1], c=color_dict[i])
			i+=1
		
		plt.savefig('figure_scatter.jpg', dip=500)
		plt.show()
		
		
	elif menu_choice == '5':
		#Classify data 
		if len(X_training) == 0:
			X_training, y_training = load_dataset()
			
		#classify Original data
		y_train_pca = np.array(y_training).copy()
		#than split dataset into training set and test set 
		X_training, y_training, X_test, y_test = split_dataset(X_training, y_training)
		#train the model with Naive Bayes Classifier 
		model = train(X_training, y_training)
		#test the model with test set and print accuracy 
		print("Computing accuracy Original Dataset")
		accuracy = test_model(model, X_test, y_test) 
		
		#classify 2 PC data
		X_train_pca2, X_pca2_transf = compute_first_PC(X_training, 2)
		y_train_pca2 = np.array(y_train_pca).copy()
		#convert from list to array	
		X_train_pca2 = np.asarray(X_train_pca2)
		y_train_pca2 = np.asarray(y_train_pca2)
		#split 2 pc data
		X_train_pca2, y_train_pca2, X_test_pca2, y_test_pca2 = split_dataset(X_train_pca2, y_train_pca2)
		#train the model with Naive Bayes Classifier 
		model = train(X_train_pca2, y_train_pca2)
		#test the model with test set and print accuracy 
		print("Computing accuracy 2PC")
		accuracy = test_model(model, X_test_pca2, y_test_pca2) 
		
		#classify 3 and 4 PC data
		X_train_pca3to4, X_pca3to4_transf = compute_PC(X_training, 3, 4) 
		y_train_pca3to4 = np.array(y_train_pca).copy()
		#convert from list to array 	
		X_train_pca3to4 = np.asarray(X_train_pca3to4)
		y_train_pca3to4 = np.asarray(y_train_pca3to4)
		#split data
		X_train_pca3to4, y_train_pca3to4, X_test_pca3to4, y_test_pca3to4 = split_dataset(X_train_pca3to4, y_train_pca3to4)
		#train the model with Naive Bayes Classifier 
		model = train(X_train_pca3to4, y_train_pca3to4)
		#test the model with test set and print accuracy 
		print("Computing accuracy 3&4PC")
		accuracy = test_model(model, X_test_pca3to4, y_test_pca3to4) 
		
	
	elif menu_choice =='6':
		sys.exit()
		
	else:
		print('Not correct choice. Try again')









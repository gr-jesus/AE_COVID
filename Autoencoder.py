import cv2
import random
import numpy as np
import SimpleITK as sitk
from os import walk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class my_autoencoder(object):
	def __init__(self,
		         data_folder='../stoic_2021/data/mha/',
		         target_size=128):
		self.data_folder=data_folder
		self.target_size=target_size
		self.dataset=np.zeros((1,4,target_size,target_size))
		self.autoencoder=None
		#self.load_partition=load_partition

	def read_data(self):
		'''
		read the data in the data_folder
		'''
		# select randomly 200 images for this case
		files=next(walk(self.data_folder), (None, None, []))[2]
		random.shuffle(files)
		for image in next(walk(self.data_folder), (None, None, []))[2]:
			aux=[]
			# open the image
			image=sitk.ReadImage(self.data_folder+image)
			# get the array
			image=sitk.GetArrayFromImage(image)
			# Reshape and normalize
			image=image[:image.shape[0]-image.shape[0]%4] # 32768~image 16 bits
			for i in image:
				aux.append(cv2.resize(i, (self.target_size,self.target_size), interpolation = cv2.INTER_AREA))
			image=np.reshape(np.array(aux), (image.shape[0],self.target_size,self.target_size))
			to_normalize=max(abs(np.min(image)), np.max(image))
			#print(np.min(image), np.max(image), to_normalize)
			image=((np.array(np.split(image, int(image.shape[0]/4.0)))/to_normalize)+1.0)/2.0 #32768.0 # 32768~image 16 bits
			# concatenate with the dataset np array
			self.dataset=np.concatenate((self.dataset, image))
			# we only use 3000 images because of hardware limitations
			if self.dataset.shape[0]>3000:
				break
		self.dataset=self.dataset[1:][:3000]
		random.shuffle(self.dataset)
		self.dataset=np.moveaxis(np.array(self.dataset),1, 3)
		return self.dataset[:2400], self.dataset[600:]
		#print(np.array(self.dataset).shape)


	def build_autoencoder(self, lr, beta_1, beta_2, epsilon):
		input_img=Input(shape=( self.target_size, self.target_size, 4), name='Input')
		x=Convolution2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv1E')(input_img)
		x=MaxPooling2D((2,2), name='Maxpool1E')(x)
		x=Convolution2D(64,(3,3), activation='relu', padding='same', name='Conv2E')(x)
		x=MaxPooling2D((2,2), name='Maxpool2E')(x)
		x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv3E')(x)
		x=MaxPooling2D((2,2), name='Maxpool3E')(x)
		x=Convolution2D(16,(3,3), activation='relu', padding='same', name='Conv4E')(x)
		encoded=MaxPooling2D((2,2), name='Maxpool4E')(x)
		x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv1D')(encoded)
		x=UpSampling2D((2,2), name='Maxpool1D')(x)
		x=Convolution2D(32,(3,3), activation='relu', padding='same', name='Conv2D')(x)
		x=UpSampling2D((2,2), name='Maxpool2D')(x)
		x=Convolution2D(64,(3,3), activation='relu', padding='same', name='Conv3D')(x)
		x=UpSampling2D((2,2), name='Maxpool3D')(x)
		x=Convolution2D(128,(3,3), activation='relu', padding='same', name='Conv4D')(x)
		x=UpSampling2D((2,2), name='Maxpool4D')(x)
		decoded=Convolution2D(4,(1,1), activation='sigmoid', name='Decode')(x)
		self.autoencoder=Model(input_img, decoded)
		opt=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
		self.autoencoder.compile(optimizer=opt, loss='mse')
		#print(autoencoder.summary())
		#return autoencoder

	def train_autoencoder(self, lr, beta_1, beta_2, epsilon, train, test):
		self.build_autoencoder(lr, beta_1, beta_2, epsilon)
		
		save_best_cb=ModelCheckpoint(filepath='best_AE.h5', monitor='val_loss', save_best_only=True)
		history=self.autoencoder.fit(train, train, validation_data=(test, test), batch_size=32, epochs=200, callbacks=[save_best_cb])
		self.autoencoder.load_weights('best_AE.h5')
		print('--',str(lr)+','+str(beta_1)+','+str(beta_2)+','+str(epsilon)+','+str(min(history.history['val_loss'])))
		plt.plot(history.history['val_loss'])
		plt.plot(history.history['loss'])
		plt.savefig('Training.png')

	def test_images(self):
		image=self.dataset[-1]
		prediction=self.autoencoder.predict(np.expand_dims(image, axis=0))[0]
		image=np.rollaxis(image,2,0)*255.0
		prediction=np.rollaxis(prediction,2,0)*255.0
		#print(image.shape, prediction.shape)
		# save the images
		for i in range(4):
			print(image[i])
			print('-----------------------------------------')
			print(prediction[i])
			cv2.imwrite('img_'+str(i)+'.png', image[i])
			cv2.imwrite('ae_'+str(i)+'.png', prediction[i])
			#cv2.imwrite('img_'+str(i)+'.png', np.clip(image[i], 0, 255))
			#cv2.imwrite('ae_'+str(i)+'.png', np.clip(prediction[i], 0, 255))
		#print(np.array(self.dataset[-1][:,:,:0]*255).shape)
		#print(np.expand_dims(self.dataset[-1], axis=0).shape)
		#print(self.autoencoder.predict(np.expand_dims(self.dataset[-1], axis=0)).shape)

	def run(self):
		#self.build_autoencoder()
		train, test = self.read_data()
		print('--,learning_rate,beta_1,beta_2,epsilon,best_val_loss')
		for lr in [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.0009]:
			for beta_1 in [0.1, 0.2, 0.04, 0.06, 0.08, 0.09, 0.99]:
				for beta_2 in [0.1, 0.2, 0.04, 0.06, 0.08, 0.09, 0.99]:
					for epsilon in [1e-7, 3e-7, 5e-7, 7e-7, 9e-7]:
						self.train_autoencoder(lr, beta_1, beta_2, epsilon, train, test)
		#self.test_images()
	def run_individual(self):
		train, test = self.read_data()
		self.train_autoencoder(0.0008, 0.99, 0.99, 1e-07, train, test)


ae=my_autoencoder()
ae.run_individual()
#model=ae.build_autoencoder()
#train, test=ae.read_data()
#model.fit(train, train, validation_data=(test, test), batch_size=32, epochs=30)
#print(np.array(train).shape)
#print(np.array(test).shape)

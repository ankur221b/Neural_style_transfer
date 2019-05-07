
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras import backend as K
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b as opt
from scipy.misc import imsave
import matplotlib.pyplot as plt
import time

height = 256
width = 256
content_weight = 25.0
style_weight = 1.0
layers_content = "block5_conv2"
layers_style = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block4_conv2"]


class StyleTransfer:

	def __init__(self,content_image,style_image):

		self.style = tf.Variable(self.load_image(style_image))
		self.content = tf.Variable(self.load_image(content_image))

		self.create_model()
		self.create_loss()

		# Important line. For using VGG16+imagenet weights we need to load
		# the session. If we don't and we do global_variables_initializer
		# the weights will be randomly initialized.
		self.sess = K.get_session()

	def create_model(self):
		
		self.combination = tf.Variable(tf.random_uniform((1,
								height,width,3)))

		input_tensor = tf.concat([self.content,self.style,
								self.combination],0)
		self.model = VGG16(input_tensor=input_tensor,
				include_top=False, weights="imagenet")


	def create_loss(self):

		outputs_dict = dict([ (layer.name, layer.output) for layer in self.model.layers ])


		self.loss, self.content_loss, self.sloss, self.style_loss= tf.Variable(0.),tf.Variable(0.),tf.Variable(0.),tf.Variable(0.)

		layer_features = outputs_dict[layers_content]
		target_image_features = layer_features[0, :, :, :]
		combination_features = layer_features[2, :, :, :]
		self.content_loss += content_weight * self.get_content_loss(target_image_features,combination_features)
		self.loss += self.content_loss

		for layer_name in layers_style:
			layer_features = outputs_dict[layer_name]
			style_reference_features = layer_features[1, :, :, :]
			combination_features = layer_features[2, :, :, :]
			self.sloss = self.get_style_loss(style_reference_features, combination_features)
			self.style_loss += (style_weight/len(layers_style)) * self.sloss
			self.loss += (style_weight/len(layers_style)) * self.sloss

		
	def get_content_loss(self,content,combination):
	    
	    return K.sum(K.square(content - combination)) 


	def get_style_loss(self,style,combination):
		
		h,w,d = style.get_shape()
		M = h.value*w.value
		N = d.value
		style_gram=self.gram_matrix(style)
		content_gram=self.gram_matrix(combination)
		loss = K.sum(K.square(style_gram-content_gram)) / ( 4.0*(N**2)*(M**2))
		return loss

	
	def train(self,epochs):
	
		train_step = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
					var_list=[self.combination],options={'maxfun':20})

		for i in range(epochs):
			curr_loss = self.sess.run(self.loss)
			c_loss = self.sess.run(self.content_loss)
			s_loss = self.sess.run(self.style_loss)
			print("Iteration {0}, Content Loss: {1}, Style Loss: {2}, Total Loss: {3}".format(i,c_loss,s_loss,curr_loss))
			train_step.minimize(session=self.sess)
			
		self.finalOutput = self.sess.run(self.combination)
		
		
			
		
		

	def saveOutput(self,name):

		out = self.finalOutput.reshape((height,width,3))
		out = np.clip(out,0,255).astype('uint8')
		imsave(name,out)		


	def gram_matrix(self,matrix):
  
  		features = K.batch_flatten(K.permute_dimensions(matrix, (2, 0, 1)))
  		gram = K.dot(features, K.transpose(features))
  		return gram


	def load_image(self,path):
  		image = Image.open(path)
  		image = image.resize((height, width))
  		image_array =  np.asarray(image, dtype="float32")
  		image_array = np.reshape(image_array, (1, height, width, 3))
	
  		return image_array


curr_time = time.time()
st = StyleTransfer("content.jpg","style.jpg")
st.train(25)
st.saveOutput("result.jpg")
print("Completed in {0} seconds".format(time.time()-curr_time))
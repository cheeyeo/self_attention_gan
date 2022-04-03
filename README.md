### Self Attention GAN

[Hands On Image Generation with Tensorflow]: https://github.com/PacktPublishing/Hands-On-Image-Generation-with-TensorFlow-2.0

Updated implementation of Self-Attention GAN based on an example from [Hands On Image Generation with Tensorflow]


Changes made in this version:

* Implented a custom model for SAGAN 

* Used custom callbacks to save model checkpoint and generate image plots at specified intervals




#### Notes On SAGAN

* Spectral Normalization

	Normalize weights by dividing by their spectral norms

	Square root of largest eigenvalue of matrix is spectral norm


	For non-square matrix we need to use SVD (Singular vector decomposition) to calculate the eigen values, which can be computationally expensive

	Use iterative algo to speed up calc

	( explain in pseudo code from book ... )

	number of iterations in spectral norm algo is a hyperparam and 5 is sufficient

	use it as kernel constraint when defining layers:
	```
	Conv2D(3, 1, kernel_constraint=SpectralNorm())
	```


* Self attention

	key, query, value

	value => representation of input features

	don't want self-attention module to inspect every single pixel as its computationally expensive

	more interested in local regions of input activations

	value has reduced dims from input features, in terms of activations map size and number of channels


	for conv layers, the channel number is reduced using a 1x1 convolution and spatial size reduced using max or average pooling


	key, query => compute importance of features of self-attention map

	to calculate output feature at location x, take query at location x and compare it with keys at all locations


	e.g. if network detected one eye in an image, it will take query and check with keys of other areas of image

	if other eye is found we can apply attention to it


	for feature 0, the eqn becomes q0xk0, q0xk1, q0xk2, ..., q0xkN-1


	vectors above normalized using softmax so their probs sum to 1.0, which is our attention score

	the score is used as a weight to perform element-wise multiplication to the value, which produces attention outputs


### TODO:

* Output loss metrics on each batch callback

* Multi-GPU distributed training...

* Implement BigGAN as per the book example


### Ref:

https://github.com/PacktPublishing/Hands-On-Image-Generation-with-TensorFlow-2.0/blob/master/Chapter08/ch8_sagan.ipynb


https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit


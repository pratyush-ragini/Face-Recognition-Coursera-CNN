
# Face Recognition using FaceNet 
Detailed report in repo.

## 1 - Encoding face images into a 128-dimensional vector 

### 1.1 - Using a ConvNet  to compute encodings

The FaceNet model takes a lot of data and a long time to train. So we load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). 


- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of m face images) as a tensor of shape (m, n_C, n_H, n_W) = (m, 3, 96, 96) 
- It outputs a matrix of shape (m, 128) that encodes each input face image into a 128-dimensional vector
<br>
By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. These encodings can then be used to compare two face images as:

<img src="images/distance_kiank.png" style="width:680px;height:250px;">
<caption><center> <u> <font color='purple'> <br> </u> <font color='purple'> By computing the distance between two encodings and thresholding, you can determine if the two pictures represent the same person</center></caption>

So, an encoding is a good one if: 
- The encodings of two images of the same person are quite similar to each other. 
- The encodings of two images of different persons are very different.

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart. 

<img src="images/triplet_comparison.png" style="width:280px;height:150px;">
<br>
<caption><center> <u> <font color='purple'> <br> </u> <font color='purple'> In the next part, we will call the pictures from left to right: Anchor (A), Positive (P), Negative (N)  </center></caption>



### 1.2 - The Triplet Loss

For an image x, we denote its encoding f(x), where f is the function computed by the neural network.

<img src="images/f_x.png" style="width:380px;height:150px;">


Training will use triplets of images (A, P, N):  

- A is an "Anchor" image--a picture of a person. 
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from our training dataset. We will write (A(i), P(i), N(i)) to denote the i-th training example. 

We have to make sure that an image A(i) of an individual is closer to the Positive P(i) than to the Negative image N(i) by at least a margin &alpha:
<br>

FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data and a lot of computation, we won't train it from scratch here. Instead, we load a previously trained model.
<br>
Here are some examples of distances between the encodings between three individuals:

<img src="images/distance_matrix.png" style="width:380px;height:200px;">
<br>
<caption><center> <u> <font color='purple'> </u> <br>  <font color='purple'> Example of distance outputs between three individuals' encodings</center></caption>

We will now use this model to perform face verification and face recognition.

## 2 - Applying the model

### 2.1 - Face Verification

A database is built first, containing one encoding vector for each person. To generate the encoding we use `img_to_encoding(image_path, model)`, which runs the forward propagation of the model on the specified image. This database maps each person's name to a 128-dimensional encoding of their face.
Then it's checked if images of same person are correctly verified, and also for different persons.

### 2.2 - Face Recognition

Now, we'd like to change the face verification system to a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who). 
This is checked on a person in the database as well as one who isn't.


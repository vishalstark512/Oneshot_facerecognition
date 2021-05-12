#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import io # Input/Output Module
import os # OS interfaces
import cv2 # OpenCV package
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from urllib import request # module for opening HTTP requests
from matplotlib import pyplot as plt # Plotting library


# <div style="width:100%; height:140px">
#     <img src="https://www.kuleuven.be/internationaal/thinktank/fotos-en-logos/ku-leuven-logo.png/image_preview" width = 300px, heigh = auto align=left>
# </div>
# 
# 
# KUL H02A5a Computer Vision: Group Assignment 1
# ---------------------------------------------------------------
# Student numbers: <span style="color:red">r0825483, r0653687, r0823967, r0820202, r0822884</span>.
# 
# The goal of this assignment is to explore more advanced techniques for constructing features that better describe objects of interest and to perform face recognition using these features. This assignment will be delivered in groups of 5 (either composed by you or randomly assigned by your TA's).
# 
# In this assignment you are a group of computer vision experts that have been invited to ECCV 2021 to do a tutorial about  "Feature representations, then and now". To prepare the tutorial you are asked to participate in a kaggle competition and to release a notebook that can be easily studied by the tutorial participants. Your target audience is: (master) students who want to get a first hands-on introduction to the techniques that you apply.
# 
# ---------------------------------------------------------------
# This notebook is structured as follows:
# 0. Data loading & Preprocessing
# 1. Feature Representations
# 2. Evaluation Metrics 
# 3. Classifiers
# 4. Experiments
# 5. FaceNet
# 6. Publishing best results
# 7. Discussion
# 
# Make sure that your notebook is **self-contained** and **fully documented**. Walk us through all steps of your code. Treat your notebook as a tutorial for students who need to get a first hands-on introduction to the techniques that you apply. Provide strong arguments for the design choices that you made and what insights you got from your experiments. Make use of the *Group assignment* forum/discussion board on Toledo if you have any questions.
# 
# Fill in your student numbers above and get to it! Good luck! 
# 
# 
# <div class="alert alert-block alert-info">
# <b>NOTE:</b> This notebook is just a example/template, feel free to adjust in any way you please! Just keep things organised and document accordingly!
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>NOTE:</b> Clearly indicate the improvements that you make!!! You can for instance use titles like: <i>3.1. Improvement: Non-linear SVM with RBF Kernel.<i>
# </div>
#     
# ---------------------------------------------------------------
# # 0. Data loading & Preprocessing
# 
# ## 0.1. Loading data
# The training set is many times smaller than the test set and this might strike you as odd, however, this is close to a real world scenario where your system might be put through daily use! In this session we will try to do the best we can with the data that we've got! 

# In[2]:


# Input data files are available in the read-only "../input/" directory

train = pd.read_csv(
    '../input/kul-h02a5a-computervision-groupassignment0/train_set.csv', index_col = 0)
train.index = train.index.rename('id')

test = pd.read_csv(
    '../input/kul-h02a5a-computervision-groupassignment0/test_set.csv', index_col = 0)
test.index = test.index.rename('id')

# Read the images as numpy arrays and store in "img" column
train['img'] = [cv2.cvtColor(np.load('../input/kul-h02a5a-computervision-groupassignment0/train/train/train_{}.npy'.format(index), allow_pickle=False), cv2.COLOR_BGR2RGB) 
                for index, row in train.iterrows()]

test['img'] = [cv2.cvtColor(np.load('../input/kul-h02a5a-computervision-groupassignment0/test/test/test_{}.npy'.format(index), allow_pickle=False), cv2.COLOR_BGR2RGB) 
                for index, row in test.iterrows()]
  

train_size, test_size = len(train),len(test)

"The training set contains {} examples, the test set contains {} examples.".format(train_size, test_size)


# *Note: this dataset is a subset of the* [*VGG face dataset*](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
# 
# ## 0.2. A first look
# Let's have a look at the data columns and class distribution.

# In[3]:


# The training set contains an identifier, name, image information and class label
train.head(1)


# In[4]:


# The test set only contains an identifier and corresponding image information.

test.head(1)


# In[5]:


# The class distribution in the training set:
train.groupby('name').agg({'img':'count', 'class': 'max'})


# Note that **Jesse is assigned the classification label 1**, and **Mila is assigned the classification label 2**. The dataset also contains 20 images of **look alikes (assigned classification label 0)** and the raw images. 
# 
# ## 0.3. Preprocess data
# ### 0.3.1 Example: HAAR face detector
# In this example we use the [HAAR feature based cascade classifiers](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) to detect faces, then the faces are resized so that they all have the same shape. If there are multiple faces in an image, we only take the first one. 
# 
# <div class="alert alert-block alert-info"> <b>NOTE:</b> You can write temporary files to <code>/kaggle/temp/</code> or <code>../../tmp</code>, but they won't be saved outside of the current session
# </div>
# 

# In[6]:


class HAARPreprocessor():
    """Preprocessing pipeline built around HAAR feature based cascade classifiers. """
    
    def __init__(self, path, face_size):
        self.face_size = face_size
        file_path = os.path.join(path, "haarcascade_frontalface_default.xml")
        if not os.path.exists(file_path): 
            if not os.path.exists(path):
                os.mkdir(path)
            self.download_model(file_path)
        
        self.classifier = cv2.CascadeClassifier(file_path)
  
    def download_model(self, path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/"            "haarcascades/haarcascade_frontalface_default.xml"
        
        with request.urlopen(url) as r, open(path, 'wb') as f:
            f.write(r.read())
            
    def detect_faces(self, img):
        """Detect all faces in an image."""
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(
            img_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
    def extract_faces(self, img):
        """Returns all faces (cropped) in an image."""
        
        faces = self.detect_faces(img)

        return [img[y:y+h, x:x+w] for (x, y, w, h) in faces]
    
    def preprocess(self, data_row):
        faces = self.extract_faces(data_row['img'])
        
        # if no faces were found, return None
        if len(faces) == 0:
            nan_img = np.empty(self.face_size + (3,))
            nan_img[:] = np.nan
            return nan_img
        
        # only return the first face
        return cv2.resize(faces[0], self.face_size, interpolation = cv2.INTER_AREA)
            
    def __call__(self, data):
        return np.stack([self.preprocess(row) for _, row in data.iterrows()]).astype(int)


# **Visualise**
# 
# Let's plot a few examples. First we will define a function to easy plot mulptiple images side by side.

# In[7]:


# parameter to play with 
FACE_SIZE = (100, 100)

def plot_image_sequence(data, n, imgs_per_row=7, cmap="brg"):
    n_rows = 1 + int(n/(imgs_per_row+1))
    n_cols = min(imgs_per_row, n)

    f,ax = plt.subplots(n_rows,n_cols, figsize=(10*n_cols,10*n_rows))
    for i in range(n):
        if n == 1:
            ax.imshow(data[i], cmap=cmap)
        elif n_rows > 1:
            ax[int(i/imgs_per_row),int(i%imgs_per_row)].imshow(data[i], cmap=cmap)
        else:
            ax[int(i%n)].imshow(data[i], cmap=cmap)
    plt.show()


# ## Next we will have to extract the features.

# In[8]:


#preprocessed data using the HAAR features
preprocessor = HAARPreprocessor(path = './', face_size=FACE_SIZE)

train_X, train_y = preprocessor(train), train['class'].values
test_X = preprocessor(test)


# In[9]:


# plot faces of Michael and Sarah

plot_image_sequence(train_X[train_y == 0], n=20, imgs_per_row=10)


# In[10]:


# plot faces of Jesse

plot_image_sequence(train_X[train_y == 1], n=30, imgs_per_row=10)


# In[11]:


# plot faces of Mila

plot_image_sequence(train_X[train_y == 2], n=30, imgs_per_row=10)


# ### 0.3.2 dlib face detector
# We can see that eventhough the HAAR preprocessing does detect faces, there are several images where it misses the face entirely. In this is the section we will suggest a different preprocessing to achieve better results. The [dlib library](http://dlib.net/), containes several functions which will allow us better extract faces from the images. Additionally we will make use of the face_utils function from the imutils library to easily convert the extracted faces to numpy arrays. First of all the libraries will need to be installed and imported.

# In[12]:


get_ipython().system('pip install dlib')
get_ipython().system('pip install imutils')
import dlib
from imutils import face_utils


# Next we will define a function to convert the bounding box from dlib into a format that OpenCV will understand.

# In[13]:


def rect_to_bb(rect):
    # Convert dlib boudning box to the format (x, y, w, h) as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # Return a tuple of (x, y, w, h)
    return (x, y, w, h)


# Here we will use the get_frontal_face_detector from the dlib library. This is a ready to use face detection algorithm, based on the histogram of oriented gradients, which we will discuss in a later section. This function will return all extracted faces with their respective labels.

# In[14]:


detector = dlib.get_frontal_face_detector()

def dlib(data,data_type = 'train'):
    #We set the size of the image
    dim = (160, 160)
    data_images=[]
    #If we are processing training data we need to keep track of the labels
    if data_type=='train':
        data_labels=[]
    #Loop over all images
    for cnt in range(0,len(data)):
        image = data['img'][cnt]
        #The large images are resized
        if image.shape[0] > 1000 and image.shape[1] > 1000:
            image = cv2.resize(image, (1000,1000), interpolation = cv2.INTER_AREA)
        #The image is converted to grey-scales
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        #Take the smallest face
        for (i, rect) in enumerate(rects):
            #Convert the bounding box to edges
            (x, y, w, h) = rect_to_bb(rect)
            #Here we copy and crop the face out of the image
            clone = image.copy()
            if(x>=0 and y>=0 and w>=0 and h>=0):
                crop_img = clone[y:y+h, x:x+w]
        #We resize the face to the correct size
        resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        data_images.append(resized)
        #And add the label to the list
        if data_type=='train':
            data_labels.append(data['class'][cnt])
    #Lastly we need to return the correct number of arrays
    if data_type=='train':
        return np.array(data_images), np.array(data_labels)
    else:
        return np.array(data_images)


# Here we extract our features once more, but this time using dlib.

# In[15]:


#preprocessed data using dlib
train_X , train_y = dlib(train, 'train')
test_X = dlib(test,'test')


# **Visualise**
# 
# Let's try again to plot a few examples.

# In[16]:


# plot faces of Michael and Sarah

plot_image_sequence(train_X[train_y == 0], n=20, imgs_per_row=10)


# In[17]:


# plot faces of Jesse

plot_image_sequence(train_X[train_y == 1], n=30, imgs_per_row=10)


# In[18]:


# plot faces of Mila

plot_image_sequence(train_X[train_y == 2], n=30, imgs_per_row=10)


# We see that in some images that the faces are not correct. This can happen if multiple faces are in an image, because the face detector cannot decide which of the faces is the correct one according to the label. When this happens we can manually edit the labels to prevent any noise in our training data.

# In[19]:


newTrainX = list()
newTrainY = list()
for index, face_pixels in enumerate(train_X[train_y == 0]):
        newTrainX.append(face_pixels)
        newTrainY.append(0)
    
for index, face_pixels in enumerate(train_X[train_y == 1]):
    if index not in [20,27,28]:
        newTrainX.append(face_pixels)
        newTrainY.append(1)
    
for index, face_pixels in enumerate(train_X[train_y == 2]):
    if index not in [9,14,15,18,22,23]:
        newTrainX.append(face_pixels)
        newTrainY.append(2)
    
TrainX = np.array(newTrainX)
TrainY = np.array(newTrainY)
print(TrainX.shape, TrainY.shape)

plot_image_sequence(TrainX, n=70, imgs_per_row=10)


# # 1. Feature Representations
# ## 1.0. Example: Identify feature extractor
# Throughout the following sections we will define several classes for the feature extractors. Here we will show a simple example to demonstrate a generic structure. Our example feature extractor doesn't actually do anything... It just returns the input:
# $$
# \forall x : f(x) = x.
# $$
# 
# It does make for a good placeholder and baseclass ;).

# In[20]:


class IdentityFeatureExtractor:
    """A simple function that returns the input"""
    
    def transform(self, X):
        return X
    
    def __call__(self, X):
        return self.transform(X)


# ## 1.1 Baseline 1: HOG feature extractor
# In this section we will explore the Histogram Oriented Gradients (HOG) feature extractor. The idea of a feature extractor is to, as the name suggests, extract important features out of an image through the orientation of gradients. These features can then be compared to other images to recognise objects or in our case faces. To this end we will create a class which we shall name HOGFeatureExtractor. Upon initialisation of the function we will set several hyper parameters which we will discuss later. 
# 
# When the extractor is called each image will be converted to grey-scale. This allows the edges to be detected more easily. The edges are then converted to polar coordinates, which results in two matrices of the same size as the input image. The first matrix contains the magnitude of the gradient in each point and the second matrix contains the direction of this gradient. The next step is the histogram part of this extractor. Here all the gradients within a cell defined by the user, are gathered into a single histogram. Each bin of this histogram represents a predefined angle, here these angles are 0 to 180 in intervals of 20 degrees. All angles are set between 0 and 180 degrees rather than to 360 degrees. This means that both positive and negative gradients are added to the same angle. (According to source these so called unsigned gradients work better in object recognition that signed gradietns. ) The magnitudes of the gradients are divided amongst their corresponding bins according to their angle.
# 
# Once the histogram is calculated for every cell, we have converted the image to sets of gradients. The strength of these gradients can depend on the lighting in the picture or shadows on the object. As we would like to determine the features of a face regardless of the light, we need to normalise the gradients. To normalise the gradients we will use block normalisation. This method determine the norm of a block of cells rather than the whole image. By only looking at a smaller region we can mitigate some of the effects of shadows. Once a block is normalised it is added as a one dimensional vector to the feature vector. These steps are then repeated for each block as it slides over the cells with a stride set by the user.
# 
# Due to the order of the operations and the nature of the HOG features, they can be rather tricky to plot. In the class we have defined a plotting function which will take the correct features to overlay over an input image such that we can inspect which features are extracted.

# In[21]:


class HOGFeatureExtractor(IdentityFeatureExtractor):
    #Upon initialising the feature extractor will set several parameters which are needed lateron
    def __init__(self, image_size = (100,100), cell_size = (3,3), block_size = (6,6), stride = (3,3)):
        self.imgSize     = image_size
        self.cellSize    = cell_size
        self.blockSize   = (np.floor(block_size[0]/self.cellSize[0]),np.floor(block_size[1]/self.cellSize[1]))
        self.blockStride = (np.floor(stride[0]/self.cellSize[0]),np.floor(stride[0]/self.cellSize[1]))
        
    #When the class is called it will turn images into HOG-feature vectors
    def __call__(self, X):
        features = []
        #Here we check if a single image has passed or an array
        if len(np.shape(X))<4:
            X = [X]
        #Loop over all images
        for i in range(len(X)):
            #First we convert the image to grey-scale
            grey = cv2.cvtColor(np.float32(X[i]), cv2.COLOR_BGR2GRAY)
            #Next we transfor the grey-scale image into a HOG-feature vector
            features.append(self.transform(grey))
        print("Extracted all HOG-feature vectors!")
        return features
      
    #The transform function will turn a single image into a vector of HOG-features
    def transform(self, img):
        #The edgdes are determined in the x and y direction
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        #The gradients are then converted to polar coordinates
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #For each cell we will record a histogram
        cells = np.zeros((int(np.floor(len(mag)/self.cellSize[0])),int(np.floor(len(mag[0])/self.cellSize[1])),9))
        #Loop over all the cells
        for i in range(len(cells[:,0,0])):
            for j in range(len(cells[0,:,0])):
                #Extracting the cell from the full image
                cellMag = mag[i*self.cellSize[0]:(i+1)*self.cellSize[0],j*self.cellSize[1]:(j+1)*self.cellSize[1]]
                cellAngle = angle[i*self.cellSize[0]:(i+1)*self.cellSize[0],j*self.cellSize[1]:(j+1)*self.cellSize[1]]
                cells[i,j,:] = self.histogram(cellMag, cellAngle)
        #The features need to be normalised per block
        feat = [] 
        #Loop over all blocks
        for i in range(int((len(cells[:,0])-self.blockSize[0])/self.blockStride[0]+1)):
            for j in range(int((len(cells[0,:])-self.blockSize[1])/self.blockStride[1]+1)):
                #Extract a block
                block = (cells[int(i*self.blockStride[0]):int(i*self.blockStride[0]+self.blockSize[0]),
                                       int(j*self.blockStride[1]):int(j*self.blockStride[1]+self.blockSize[1]),:])
                #Turn the block into a 1D vector
                block = np.ravel(block)
                #In large areas where there are no gradients we just keep 0
                if np.linalg.norm(block)==0:
                    feat.append(np.zeros(np.shape(block)))
                #Otherwise the block is normalised
                else:
                    feat.append(block/np.linalg.norm(block))
        #All blocks are reshaped into a single 1D and returned
        feat = np.ravel(feat)
        return feat 
    
    #The histogram function takes a cell and builds a histogram for the oriented gradients in this cell
    def histogram(self, cellMag, cellAngle):
        Hist = np.zeros(9)
        #Loop over all elements in the cell
        for i in range(len(cellMag[0,:])):
            for j in range(len(cellMag[:,0])):
                #First we find the lowest bin close to the angle
                ind = int(np.floor(cellAngle[i,j]%180/20))
                #The contribution to this bin is calculated and added
                scale = 1 - (cellAngle[i,j]-ind*20)/20
                Hist[ind] =+ scale*cellMag[i,j]
                #The contribution to the following bin is calculated
                scale = 1 - ((ind+1)*20-cellAngle[i,j])/20
                #If the next bin would be 180 degrees, the contribution is added to 0 degrees
                if ind == len(Hist)-1:
                    Hist[0] =+ scale*cellMag[i,j]
                else:
                    Hist[ind] =+ scale*cellMag[i,j]
        return Hist
                     
    #The plot function is used to show the HOG features of an image
    def plot(self, feat, img = None):
        #If a source image is give the HOG features will be plotted over the image
        if img.any():
            plt.imshow(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY), cmap = 'gray')
        #For ease of use the feature vector is reshaped to a vector of histograms
        feat = feat.reshape(int(len(feat)/9),9)
        #We will need the angles again as well
        angles = np.float64([0, 20, 40, 60, 80,100,120,140,160])
        #Looping over each cell in the image
        for i in range(int(self.imgSize[0]/self.cellSize[0])):
            for j in range(int(self.imgSize[1]/self.cellSize[1])):
                #In the first block of rows the indices are counted different
                if i < self.blockSize[1]:
                    #Similarly in the first block of columns the indices are also counted different
                    if j < self.blockSize[0]:
                       currentHist = i*self.cellSize[0]+j
                    else:
                       currentHist = (self.blockSize[0]*self.blockSize[1]*(np.floor((j-self.blockSize[0])/self.blockStride[0])+1)
                           +(j-self.blockSize[0])%self.blockStride[0]+(self.blockSize[0]-self.blockStride[0])+i*self.blockSize[0])
                else:
                    last = ((np.floor((self.imgSize[0]/self.cellSize[0]-self.blockSize[0])/self.blockStride[0])+1)*self.blockSize[0]*self.blockSize[1]
                           *(np.floor((i-self.blockSize[1])/self.blockStride[1])+1))
                    if j < self.blockSize[0]:
                       currentHist = last+((i-self.blockSize[1])%self.blockStride[1]+(self.blockSize[1]-self.blockStride[1]))*self.blockSize[0]+j
                    else:
                       currentHist = (last+((i-self.blockSize[1])%self.blockStride[1]+(self.blockSize[1]-self.blockStride[1]))*self.blockSize[0]
                                    +(self.blockSize[0]*self.blockSize[1]*(np.floor((j-self.blockSize[0])/self.blockStride[0])+1)
                                    +(j-self.blockSize[0])%self.blockStride[0]+(self.blockSize[0]-self.blockStride[0])))
                #We set the origin of each histogram at the centre of the corresponding cell
                originX = []
                originY = []
                for l in range(9):
                    originX.append((i+0.5)*self.cellSize[0])
                    originY.append((j+0.5)*self.cellSize[1])
                #Convert back to carthesian
                x, y = cv2.polarToCart(feat[int(currentHist)], angles, angleInDegrees=True)
                #Plot all positive
                plt.quiver(originY,originX, y, x, scale_units='xy', scale=1.5/self.cellSize[0])
                #Plot all negative
                plt.quiver(originY,originX,-y,-x, scale_units='xy', scale=1.5/self.cellSize[0])
        #If no source image was given the HOG features will be plotted upside down, so this fixes it
        if not img.any():
            plt.gca().invert_yaxis()


# With this we have created our HOG-feature extractor, but now we still need to initialise an instance of our class. Here we will pass the input parameters. These parameters are all counted in number of pixels. This only needs to be done once, after which we can pass our images to the extractor.

# In[22]:


image_size = (160,160)
cell_size  = (10,10)
block_size = (20,20)
stride = (10,10)

hog  = HOGFeatureExtractor(image_size, cell_size, block_size, stride )
feat = hog(train_X)


# Let's have a look at some examples. First we will increase the figure size, which will make the HOG-features clearer. To see a couple different images we will loop over 5 images. To plot the HOG features we will use the plotting function we included in the extractor class. Inorder to see how the features relate to the original image, we will pass the image as an arguement as well.

# In[23]:


fig = plt.figure(figsize=(50, 50))
for i in range(5):
    plt.subplot(1,5,i+1)
    hog.plot(feat[i],train_X[i])
plt.show()


# We can see that the feature extractor finds the contours of the faces.The actual feature vector will contain much more information than we can display onto these images as the blocks in the normalising step overlap considerably. We can see immediately that if we were to roate the images by say 90 degrees the vectors will look completely different, making this method not rotation invariant. Similarly if the image would be of a different scale a single cell might describe a whole nose rather than a part of it, making this method also sensitive to scale. This problem however, is caught to some degree by our face detector which extracts all faces to the same size.
# 
# ### 1.1.1. Scale-Invariant Feature Transform 
# Another method to describe faces is through Scale-Invariant Feature Transform (SIFT). We will not discuss the method in detail in order to not make this already lenghty notebook any longer, but for the sake of completeness, we will briefly discuss the outline of the method here. This method will search for keypoints in the training images and compare them to keypoints in new images. Opencv provides this method, but to easily apply it to all of our images we will define a new function.

# In[24]:


def SIFT(X ):
    keypoints = []
    #Here we check if a single image has passed or an array
    if len(np.shape(X))<4:
        X = [X]
    #Loop over all images
    for i in range(len(X)):
        #The images are converted to grey-scale and the correct format
        grey= cv2.cvtColor(np.float32(X[i]), cv2.COLOR_BGR2GRAY).astype('uint8')
        #Here we extract the keypoints
        kp, des = cv2.SIFT_create().detectAndCompute(grey,None)
        keypoints.append(kp)
    return keypoints


# With the extractor defined, let us extract the keypoints and look at a few examples again.

# In[25]:


kp  = SIFT(train_X)
fig = plt.figure(figsize=(50, 50))
for i in range(5):
    plt.subplot(1,5,i+1)
    #Here we also need to convert the image to a format opencv understands
    img = cv2.cvtColor(np.float32(train_X[i]), cv2.COLOR_BGR2GRAY).astype('uint8')
    img = cv2.drawKeypoints(img,kp[i],img)
    plt.imshow(img, cmap = 'gray')
plt.show()


# These keypoints can then be combined into a dictionary of descriptions following a Bag of Words model. With this dictionary each face can be described by a set of "words". These words can then be passed to a classifier to recognise the different faces in these images. As mentioned before, we will not discuss this method in depth.

# ### 1.1.2. t-SNE Plots
# The feature vectors have a very high dimension (usually in the thousands) and this is rather difficult (actually impossible) to visualise. For this reason we will try to reduce the number of dimensions. We can do this through t-distributed stochastic neighbor embedding (t-SNE). In short, this method will construct probabilities such that similar vectors will have high probabilities. This method however, does not work well when we reduce the number of dimensions so drastically. This is why it is suggested [(scikit documentation)](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to first reduce the dimensions using principal component analysis. Though even when applying this method it is near impossible to represent this high number of dimensions in any meaningful way.

# In[26]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Finding the 50 most important components from the  "+str(len(feat[0]))+" dimensions.")
pca = PCA(n_components=50)
pca_res = pca.fit_transform(feat)
print("Compressing  50 principle components into 2 dimensions.")
feat_embedded = TSNE(n_components=2,init='pca').fit_transform(pca_res)
plt.scatter(feat_embedded[train_y == 0][:,0],feat_embedded[train_y == 0][:,1], color='r', label="0")
plt.scatter(feat_embedded[train_y == 1][:,0],feat_embedded[train_y == 1][:,1], color='g', label="1")
plt.scatter(feat_embedded[train_y == 2][:,0],feat_embedded[train_y == 2][:,1], color='b', label="2")
plt.legend()
plt.show()


# ## 1.2. Baseline 2: PCA feature extractor
# An important and interesting technique for face image representation is based upon principal component analysis, or PCA. It is a dimensionality reduction method that given a space of D dimensions finds k principal components, or vectors (with k << D) that can be used to remap the input data into a new smaller space still accounting for a good amount of data variance, or information. In this context however, such vectors can be considered "faces", in literature called EigenFaces. A face can be described as linear combination of these eigenfaces and the associated coefficients determine the vectorization of the face itself.
# It is possible to argue that SVD, another dimensionality reduction method, is better performing than PCA because it doesn't require matrix multiplication. However, by centering our input data to 0 (subtracting the average face) PCA behaves as SVD. Indeed, sklearn PCA implementation uses SVD to return the principal components.
# In this section we will show how to create such space, extract these features for each image and also how we ca reconstruct the input image as linear combination of the chosen eigenfaces. 

# In[27]:


class PCAFeatureExtractor(IdentityFeatureExtractor):
    
    def __init__(self, size, n_components=None):
        self.n_components = n_components
        
        # initialize PCA with the specified number of components
        self.pca = PCA(n_components=n_components)
        
        self.mean_face = None
        self.isTrain = True
        self.size = size
        
    def __preprocess(self, X):
        # to grayscale
        X = np.array(np.mean(X, -1))
        
        # reshape to img_count * width x height
        d = X.shape[1] * X.shape[2]
        X = np.reshape(X, (X.shape[0], d))
        
        # make sure that each value is within [0, 255]
        X = np.clip(X, 0, 255)
        
        # if the mean face has not been computed, do so as the 
        # numerical mean of each input image
        if self.mean_face is None:
            self.mean_face = np.mean(X, axis=0)
            
        # subtract the mean face to each input face
        X -= self.mean_face
        
        return X
    
    # plot mean face as image
    def plot_mean_face(self):
        plt.title("Mean face")
        plt.imshow(np.reshape(self.mean_face, self.size), cmap="gray")
        plt.show()
    
    # plot all the principal components as images --> eigenfaces
    def plot_eigenfaces(self, imgs_per_row):
        data = []
        for comp in self.pca.components_:
            data.append(np.reshape(comp, self.size))
        
        plot_image_sequence(data, len(data), imgs_per_row=7, cmap="gray")
        
    # plot the cumulative explained variace ratio and return the first
    # number of components that ensure at least min_var_exp of variance 
    # explained
    def plot_explained_var(self, min_var_exp=0.9):
        cumsum = self.pca.explained_variance_ratio_.cumsum()
        plt.title("Cumulative explained variance over components",size=15)
        plt.xlabel('Number of components', size=12)
        plt.ylabel('Cumulative explained variance', size=12)
        plt.plot(cumsum)
        plt.show()
        
        for i, c in enumerate(cumsum, 1):
            if c >= min_var_exp:
                return i
    
    # reconstruct image i by using eigenfaces from 1 to n_components
    def image_reconstruction(self, i, x_trans):
        comps = self.pca.components_
        features = x_trans[i]

        rec_steps = []
        for n_comps in range(len(self.pca.components_)):
            rec = np.dot(features[:n_comps+1], comps[:n_comps+1])
            rec_steps.append(np.reshape(rec, self.size))

        plot_image_sequence(rec_steps, len(rec_steps), cmap="gray")
        
    # transform the input images in n_components features by using PCA
    def transform(self, X):
        X = self.__preprocess(X)
        
        # if we are in train mode we fit the PCA model on the images
        if self.isTrain:
            isTrain = False
            self.pca.fit(X)
            
        # otherwise we transform only
        return self.pca.transform(X)
        
    def inverse_transform(self, X):
        # from the eigen vectors reconstruct the image
        return self.pca.inverse_transform(X)
     


# Note that before apllying PCA to the images, we apply so,e preprocessing. In particular we do the following:
# * grayscale conversion: each pixel is associated with one value only and not 3 (RGB channels)
# * reshaping: each image is a one dimensional vector and not a matrix. This is required for feeding skitlearn the right type of data
# * value clipping: each value of such vector is between 0 and 255. It is done just to be sure that we are dealing with correct data.
# * data normalization: we first compute the mean face as vectorwise mean and then we subtract it from each input image in order to center them to 0. This final step ensures mathematical constrains for using PCA correctly and in the more efficient way 

# In order to determine a good number of eigenfaces to use, we can plot the cumulative explained variance over the number of original components and pick a component count that accounts for the wanted percentage of variance, in this case 90%.

# In[28]:


pca_ = PCAFeatureExtractor(image_size)
x_trans = pca_.transform(train_X)
best_comp_count = pca_.plot_explained_var(min_var_exp=0.9)
print("Best number of components:", best_comp_count)


# We could also find an appropriate number of components by looking at the average reconstruction loss. This loss defines the distance between the image reconstructed by using the specified number of eigenfaces and the original image itself. An ideal value is close to 0 as it means close to perfect reconstruction from the eigenfaces. In reality this doesn't happen, especially if the number of eigenfaces is far from the original number of components. 

# In[29]:


# best number of components based on reconstruction loss
def comps_rec_loss():
    min_loss = 1000
    best_j = -1
    losses = []
    for j in range(80):
        pca_ = PCAFeatureExtractor(j + 1)
        x_trans = pca_.transform(train_X)
        reconv = pca_.inverse_transform(x_trans)
        vals = []
        for i in range(train_X.shape[0]):
            if np.mean(np.reshape(np.mean(train_X[i], -1), -1) - reconv[i]) >= 0:
                vals.append(np.mean(np.reshape(np.mean(train_X[i], -1), -1) - reconv[i]))
            else:
                continue
        loss = np.mean(vals) 
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            best_j = j + 1

    print("Best number of components:", best_j)
    print("Reconstruction loss: %.3f" % min_loss)
    plt.plot(np.arange(80), losses)
    plt.show()

comps_rec_loss()


# Such method suggests only a few principal components. However, since the difference of reconstruction loss is very small across all components (the difference is in 1e-6 range), we can stick to the previously found value as it explains a good amount of variance without causing any difference in terms of reconstruction loss. In addition, the found value still reduces the problem dimensions making the computations faster. 

# ### 1.2.1. Eigenface Plots
# Now that we have the right amount of features, capable to explain more than 90% of the variance of our data, we can plot the associated eigen vectors. As sasid, these vectors can be shown as face images, also known as eigenfaces. The first image is the average face extracted from the train set (and actually subtracted from each train set image). The other images are the sorted eigenfaces that the model has produced. 

# In[30]:


pca_ = PCAFeatureExtractor(image_size, best_comp_count)
x_trans = pca_.transform(train_X)
pca_.plot_mean_face()
pca_.plot_eigenfaces(10)


# The found eigenfaces can be used to reconstruct the original image by multiplying them with the PCA representation of the image itself. It is interesting to see how the reconstruction done in this way improves the more eigenfaces we use. Below we can see the original image and the reconstructed ones that use more and more eigenfaces.

# In[31]:


# image reconstruction
image_num = 2
plt.imshow(np.mean(train_X[image_num], -1), cmap="gray")
plt.show()
pca_.image_reconstruction(image_num, x_trans)


# ### 1.2.2. Feature Space Plots
# After all, images are transformed into vectors, so by using the first two dimensions of these vectors (associated to the first two eigenfaces) we can plot them into a 2D graph in order to better see if the model is able to discriminate between different classes, keeping close related images nonetheless. 

# In[32]:


from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

# take the first 2 features for each transformed image
coords = x_trans[:, :2]
fig, ax = plt.subplots(figsize=(20, 20))

# plot the original image based on the position defined by the first
# two features of the transformed images
for coords, img in zip(coords, train_X):
    imscatter(coords[0], coords[1], img, ax=ax, zoom = 0.6)
    ax.scatter(coords[0], coords[1])
    
plt.title('Projecting original images in 2D eigenspace', fontsize=40)       
plt.xlabel("1st eigenface", fontsize=20)
plt.ylabel("2nd eigenface", fontsize=20)

plt.show()


# Such 2D visualization unfortunately doesn't show good discriminative properties since images of different classes are blended together. However, it shows some robustness as female faces are generally put towards the top of the representation while male faces towards the bottom. In addition, non-face images (befere we removed them from the input data) were clearly separated from the others. We can still see this by taking a look at the right most images, which, due to facial expressions or items like glasses, are recognized as the model to be slightly different from the rest of the data. We have to keep in mind that this representation considers only the first two dimensions of our vectorized images, so we cannot ensure that what we see here defines exactly what happens under the hood. 

# # 2. Evaluation Metrics
# As an evaluation metric we take the accuracy. Informally, accuracy is the proportion of correct predictions over the total amount of predictions. It is used a lot in classification but can give a wrong impression when dealing with unbalanced classes. As we already know that each class has roughly the same amount of examples in our traning set this will not be a problem here, though it is important to keep these limitations in mind.

# In[33]:


from sklearn.metrics import accuracy_score


# # 3. Classifiers
# ## 3.0. Example: The *'not so smart'* classifier
# This random classifier is not very complicated. It makes predictions at random, based on the distribution obseved in the training set. **It thus assumes** that the class labels of the test set will be distributed similarly to the training set. Clearly this classifier will not be useful in any real face detection, but a simple example can be useful to easily see the structure of a classifier. 

# In[34]:


class RandomClassificationModel:
    """Random classifier, draws a random sample based on class distribution observed 
    during training."""
    
    def fit(self, X, y):
        """Adjusts the class ratio instance variable to the one observed in y. 

        Parameters
        ----------
        X : tensor
            Training set
        y : array
            Training set labels

        Returns
        -------
        self : RandomClassificationModel
        """
        
        self.classes, self.class_ratio = np.unique(y, return_counts=True)
        self.class_ratio = self.class_ratio / self.class_ratio.sum()
        return self
        
    def predict(self, X):
        """Samples labels for the input data. 

        Parameters
        ----------
        X : tensor
            dataset
            
        Returns
        -------
        y_star : array
            'Predicted' labels
        """

        np.random.seed(0)
        X = np.array(X)
        return np.random.choice(self.classes, size = X.shape[0], p=self.class_ratio)
    
    def __call__(self, X):
        return self.predict(X)
    


# ## 3.1. Random Forest Classifier
# Now that we have our feature vectors for each image we can go ahead and train a classification model that can make good use of such representation. This classifier is actually a combination of two inner classifiers, both random forests. Each forest has to learn how to correctly classify class 1 or class 2. Anything else that does not belong to their assigned class should be returned as 0. Then for each classification we compare the results in the following way:
# * if forest1 says that it's 1 and forest2 says that it is something that doesn't know we assign 1
# * if the opposite happens (forest1 = 0 and forest2 = 2) we assign 2
# * in any other case, either they both agree on 0 or they disagree, we return 0

# In[35]:


from sklearn.ensemble import RandomForestClassifier
class RandomForest:
    
    def __init__(self):
        # init classifiers
        self.clf1 = RandomForestClassifier(max_depth=3, random_state=0) # binary classifier. Correct class 1 else 0
        self.clf2 = RandomForestClassifier(max_depth=3, random_state=0) # binary classifier. Correct class 2 else 0
    
    def fit(self, X, y):
        y_class1 = [v if v == 1 else 0 for v in y] # labels of the first forest are set to either 1 or 0
        y_class2 = [v if v == 2 else 0 for v in y] # here are instead set to 2 or 0
        
        self.clf1.fit(X, y_class1)
        self.clf2.fit(X, y_class2)
        
    def predict(self, X):
        pred1 = self.clf1.predict(X)
        pred2 = self.clf2.predict(X)
        
        preds = []
        for p1, p2 in zip(pred1, pred2):
            if p1 == 0 and p2 == 2:
                preds.append(2)
            elif p1 == 1 and p2 == 0:
                preds.append(1)
            else:
                preds.append(0)
        return preds
    
    def __call__(self, X):
        return self.predict(X)
    


# # 4. Experiments
# <div class="alert alert-block alert-info"> <b>NOTE:</b> Do <i>NOT</i> use this section to keep track of every little change you make in your code! Instead, highlight the most important findings and the major (best) pipelines that you've discovered.  
# </div>
# <br>
# In this section we will have a look at how well each of these feature extractors performs. As we are not given the labels for the test set, we cannot test the accuracy on unseen data. Instead we will pass the training data through the classifier. This will end up giving us a very optimistic view of the used methods as it already knows these exact examples, but nevertheless it can give us a rough idea of the different methods.
# 
# ## 4.1. HOG Features
# First let us test the accuracy of the HOG-feature extractor with the RandomClassificationModel:

# In[36]:


# Define the extractor and the classifier
feature_extractor =  HOGFeatureExtractor(image_size=np.shape(train_X[0]), cell_size=(10,10), block_size=(20,20), stride=(10,10))
classifier = RandomClassificationModel()

# Train the model on the features
classifier.fit(feature_extractor(train_X), train_y)

# Evaluate performance of the model on the training set
train_y_star = classifier.predict(feature_extractor(train_X)) # evaluation on train set

print("Accuracy on train set: %.3f" % accuracy_score(train_y, train_y_star))


# Unsurpisingly this classifier performs very poorly in most cases seeing as we simply guess the class at random. So let us try again with the RandomForestClassifier:

# In[37]:


# Define the extractor and the classifier
feature_extractor =  HOGFeatureExtractor(image_size=np.shape(train_X[0]), cell_size=(10,10), block_size=(20,20), stride=(10,10))
classifier = RandomForest()

# Train the model on the features
classifier.fit(feature_extractor(train_X), train_y)

# Model/final pipeline
model = lambda X: classifier(feature_extractor(X))

# Evaluate performance of the model on the training set
train_y_star = model(train_X)

print("Accuracy on train set: %.3f" % accuracy_score(train_y, train_y_star))


# This perfect accuracy on the otherhand does not mean it will always predict the correct face, rather it can perfectly recognise the faces in this training set. This almost always points towards the fact that it overfits the training data. Unfortunately we cannot derive any information about how well this generalises without a test set.
# 
# To creat the predictions on the test set we can simply pass it as follows:

# In[38]:


# predict the labels for the test set 
test_y_star = model(test_X)


# ## 4.1. PCA Features
# Next we look at the features extraced from the PCA method:

# In[ ]:


# Define the extractor and the classifier
feature_extractor = PCAFeatureExtractor(image_size, best_comp_count) 
classifier = RandomForest()

# Train the model on the features
classifier.fit(feature_extractor(train_X), train_y)

# Model/final pipeline
model = lambda X: classifier(feature_extractor(X))

# Evaluate performance of the model on the training set
train_y_star = model(train_X)

"The performance on the training set is {:.2f}. This however, does not tell us much about the actual performance (generalisability).".format(
    accuracy_score(train_y, train_y_star))


# Here as well we can predict the labels for our test set.

# In[ ]:


# predict the labels for the test set 
test_y_star = model(test_X)


# # 5. FaceNet
# To really push the limits of face detection we will look at some state of the art methods. Modern day face extraction techniques have made use of Deep Convolution Networks. As we all know that features created by modern deep learning frameworks are really better than most handcrafted features. We checked 4 deep learning models namely, FaceNet (Google), DeepFace (Facebook), VGGFace (Oxford) and OpenFace (CMU). Out of these 4 models  [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)  was giving us the best result. In general, FaceNet gives better result than all the other 3 models.
# 
# <div style="width:100%; height:500px">
#     <img src="https://d3i71xaburhd42.cloudfront.net/b2b0a001cf247691b3b130efa31f50ceb3ff758f/3-TableII-1.png" width = 600px, heigh = auto align=left>
# </div>
# 
# 
# FaceNet uses the following architecture:
# 
# <div style="width:100%; height:500px">
#     <img src="https://developer.ridgerun.com/wiki/images/thumb/e/eb/Googlenet.png/1800px-Googlenet.png" width = 1200px, heigh = auto align=left>
# </div>
# 
# 
# FaceNet uses inception module in blocks to reduce the number of trainable parameters. This model takes RGB images of 160x160 and generates an embedding of size 128 for an image. For this implementation we will need a couple extra functions.

# In[39]:


## from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.models import load_model
import tensorflow as tf

get_ipython().system('pip install cmake')
get_ipython().system('pip install dlib')
import dlib


# ## 5.1 Improving the preprocessing
# DLIB is a widely used model to detecting faces. In our experiments we found that dlib produces better results than HAAR, though we noticed some improvements could still be made:
# * If rectangle face bounds move out of image, we take the whole images instead of the face cropping. It is implemented as follows:
#   * if (x>=0 and y>=0 and w>=0 and h>=0):
#      * crop_img = clone[y:y+h, x:x+w]
#   * else:
#      * crop_img = clone.copy()
# * For test images, instead of saving one face per image we are saving all the faces for prediction.
# * Rather than a HOG based detector, we can use a CNN based detector.
# As these improvements are tailored to optimise for use with FaceNet, we will define a new corrected face detection. 

# In[42]:


detector = dlib.cnn_face_detection_model_v1("../input/pretrained-models-faces/mmod_human_face_detector.dat")

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.rect.left()
    y = rect.rect.top()
    w = rect.rect.right() - x
    h = rect.rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def dlib_corrected(data, data_type = 'train'):
    #We set the size of the image
    dim = (160, 160)
    data_images=[]
    #If we are processing training data we need to keep track of the labels
    if data_type=='train':
        data_labels=[]
    #Loop over all images
    for cnt in range(0,len(data)):
        image = data['img'][cnt]
        #The large images are resized
        if image.shape[0] > 1000 and image.shape[1] > 1000:
            image = cv2.resize(image, (1000,1000), interpolation = cv2.INTER_AREA)
        #The image is converted to grey-scales
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Detect the faces
        rects = detector(gray, 1)
        sub_images_data = []
        #Loop over all faces in the image
        for (i, rect) in enumerate(rects):
            #Convert the bounding box to edges
            (x, y, w, h) = rect_to_bb(rect)
            #Here we copy and crop the face out of the image
            clone = image.copy()
            if(x>=0 and y>=0 and w>=0 and h>=0):
                crop_img = clone[y:y+h, x:x+w]
            else:
                crop_img = clone.copy()
            #We resize the face to the correct size
            rgbImg = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
            #In the test set we keep track of all faces in an image
            if data_type == 'train':
                sub_images_data = rgbImg.copy()
            else:
                sub_images_data.append(rgbImg)
        #If no face is detected in the image we will add a NaN
        if(len(rects)==0):
            if data_type == 'train':
                sub_images_data = np.empty(dim + (3,))
                sub_images_data[:] = np.nan
            if data_type=='test':
                nan_images_data = np.empty(dim + (3,))
                nan_images_data[:] = np.nan
                sub_images_data.append(nan_images_data)
        #Here we add the the image(s) to the list we will return
        data_images.append(sub_images_data)
        #And add the label to the list
        if data_type=='train':
            data_labels.append(data['class'][cnt])
    #Lastly we need to return the correct number of arrays
    if data_type=='train':
        return np.array(data_images), np.array(data_labels)
    else:
        return np.array(data_images)


# Once again we will need to extract the features.

# In[43]:


corrected_test_X  = dlib_corrected(test,'test')
train_X , train_y = dlib_corrected(train, 'train')
train_X  = train_X.astype(int)


# We found that few faces were still not good to be used as training data. These "bad" faces were again manually found and removed from the training data. This reduces the number of positive training examples we have, but will also reduce the noise in our training data. 

# In[44]:


newTrainX = list()
newTrainY = list()
for index, face_pixels in enumerate(train_X[train_y == 0]):
        newTrainX.append(face_pixels)
        newTrainY.append(0)
    
for index, face_pixels in enumerate(train_X[train_y == 1]):
    if index not in [8,14,18,19,21,24]:
        newTrainX.append(face_pixels)
        newTrainY.append(1)
    
for index, face_pixels in enumerate(train_X[train_y == 2]):
    if index not in [0,7,22,23,24,27,28,14]:
        newTrainX.append(face_pixels)
        newTrainY.append(2)
    
newTrainX = np.array(newTrainX)
newTrainY = np.array(newTrainY)
print(newTrainX.shape, newTrainY.shape)


# We can now plot the manually edited training set.

# In[45]:


plot_image_sequence(newTrainX, n=newTrainY.shape[0], imgs_per_row=10)


# ## 5.2 Classifiers, again
# With the faces extracted as well as possible, we can pass these images through FaceNet.

# In[46]:


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

model = load_model('../input/pretrained-models-faces/facenet_keras.h5')

svmtrainX = []
for index, face_pixels in enumerate(newTrainX):
    embedding = get_embedding(model, face_pixels)
    svmtrainX.append(embedding)
    
svmtrainX = np.array(svmtrainX)
svmtrainY = newTrainY.copy()
svmtrainX.shape, svmtrainY.shape


# To decide on the optimal classifier we should inspect the extracted features. For this we will use PCA once more to visualise our results:

# In[47]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(svmtrainX)
principalDf = pd.DataFrame(data = principalComponents,
             columns = ['principal component 1', 'principal component 2'])

target = pd.DataFrame(svmtrainY)
finalDf = pd.concat([principalDf, target], axis = 1)

finalDf.columns = ['PC1','PC2', 'Target']
finalDf.head()


# In[48]:


fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c = color, s = 50)
ax.legend(targets)

for i in range(0, len(finalDf)):
    ax.annotate(i, (finalDf['PC1'][i], finalDf['PC2'][i]))
    
ax.grid()


# After plotting the PCA data for the extracted features from FaceNet model, we figured out that training data was linearly separable with SVM. Since this gives us buffer zone, it is able to perform better on test data even with less training data. 
# 
# 
# <div style="width:100%; height:450px">
#     <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png" width = 400px, heigh = auto align=left>
# </div>
#  
# We tried several other models and found they do not perform as well for the testing data because of the overfitting nature of classifiers with limited train data. The closest thing to SVM classifier was Naïve Bayes. Other models we tested included KNN, Decision Tree, Random forest and Neural nets. 
# 

# In[49]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

linear_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma=0.01, probability =True))
linear_model.fit(svmtrainX, svmtrainY)


# The kernel, C and gamma values are tuned according the visualization of PCA output and submission score to optimize the accuracy on test data.

# Now we can proceed to classifying our images. Firstly, we noted that a lot images had multiple faces in it. We extracted all these faces and made prediction on each face for that image. 
# * If an image had multiple faces and none of them has Jesse's or Mila's face, it is classified as the third class. 
# * If on the other hand it contains Jesse’s or Mila’s face, we chose that class rather than the third class. 
# * If both Jesse and Mila are predicted in an image then we choose the class based on their prediction confidence.

# In[50]:


predicitons=[]
for i in corrected_test_X:    
    flag=0
    if(len(i)==1):
        embedding = get_embedding(model, i[0])
        tmp_output = linear_model.predict([embedding])
        predicitons.append(tmp_output[0])
    else:
        tmp_sub_pred = []
        tmp_sub_prob = []
        for j in i:
            j= j.astype(int)
            embedding = get_embedding(model, j)
            tmp_output = linear_model.predict([embedding])
            tmp_sub_pred.append(tmp_output[0])
            tmp_output_prob = linear_model.predict_log_proba([embedding])
            tmp_sub_prob.append(np.max(tmp_output_prob[0]))
            
        if 1 in tmp_sub_pred and 2 in tmp_sub_pred:
            index_1 = np.where(np.array(tmp_sub_pred)==1)[0][0]
            index_2 = np.where(np.array(tmp_sub_pred)==2)[0][0]
            if(tmp_sub_prob[index_1] > tmp_sub_prob[index_2] ):
                predicitons.append(1)
            else:
                predicitons.append(2)
        elif 1 not in tmp_sub_pred and 2 not in tmp_sub_pred:
            predicitons.append(0)
        elif 1 in tmp_sub_pred and 2 not in tmp_sub_pred:
            predicitons.append(1)
        elif 1 not in tmp_sub_pred and 2 in tmp_sub_pred:
            predicitons.append(2)


# # 6. Publishing best results

# To save the best results we will export each prediction as a csv file.

# In[51]:


submission = test.copy().drop('img', axis = 1)
submission['class'] = predicitons

submission


# In[52]:


submission.to_csv('submission.csv')


# # 7. Discussion
# With this we will conclude this overview of facial recognition methods.
# 
# In summary we did the following: 
# * The first step in facial recognition is the detection of faces. For this we have seen the light weight HAAR detector and the heavier, but more accurate, detectors in the dlib library.
# * Regardless of the detector it is always a good practice to inspect the training data for any possible problems.
# * We saw the HOG feature detector which excells at detecting shapes but is sensitive to rotations and scale.
# * The PCA which shows slightly better results than HOG feature detector but it is still suboptimal if compared with more advanced neural network based architectures. With eigenfaces we couldn't go past 55% on Kaggle and therefore we decided to move to something more advanced.
# * Lastly we looked at state of the art facial recognition with FaceNet. This deep neural network recognises faces with  extremely high accuracy.
# 
# Deep neaural networks are able to extract more meaningful features than machine learning models. The downfall of these big networks is however the need for a huge amount of data. We managed to cope with this issue by using a pretrained model, a model that has been trained on a way bigger dataset in order to retain knowledge on how to encode face images, that we then used for our purposes in this competition. We spent a lot of time improving the quality of the few images we had to work with but if we had more time we might have improved their quality a bit more so that we could have gotten even better results. In addition, fine tuning the classifier of our last model even more would have definitely helped out more. 
# In general, what we discovered however is that what matters most is the quality of the vectorization of each image. A model such us the one we have exploited has fine grained capabilities on vectorizing the images in a meaningful way while the initial methods,

# Week 1

## Computer vision

- algorithms for a high-level understanding of images/videos
- simulate/replicate processes that biological visual systems can do
- include AI, ML, physics, neuroscience, psychology

## Image

- a pattern formed by light falling on a photosensitive surface
- light is reflected off of objects in the world
- 2D projection of a 3D world

## Lenses

- focus light rays onto a single point (F) at a distance (f) beyond the lens
- aperture diameter (D) restricts the range of rays

## Pinhole camera model

- $$x_{i} = f\frac{x}{z}$$
- $$y_{i} = f\frac{y}{z}$$
- z = distance to pt in world (big distance)
- f = distance from pinhole to image surface (small distance)

## Depth of field

- aperture size controls the depth of field
- smaller aperture, greater range of depth in focus

## Digital image

- a tensor (3D dimensional array of values)
- width x height x channel
- 3 channels (RGB)
- 1 channel (grayscale)
- pixel = smallest unit of an image
  - grayscale image: a pixel is a grayscale value
  - colour image: a pixel is a 1x3 vector

## Common file formats

- ## Lossy compression

  - JPEG

- ## Lossless compression

  - PNG, BMP, GIF, TIF

## Image manipulation

- crop

  - extract a subset of the image array (no resampling needed)

- resize

  - change dimensions of image array (resampling needed)

- ## Image scaling

- ## Resampling methods

  - Nearest-neighbour

    - closest value to sample point
      - simple, preserve hard edges
      - smooth curves may be blockly/distorted

  - Bilinear

    - weighted average of 4 pixels around sample point
      - smoother curves, but blurs hard edges
      - slower to compute

  - Others
    - bicubic, lanczos

# Week 2

## Spatial filtering

## Convolution

- operations that apply a linear filter to an image

- ### Pixel operator

  - compute output value at each pixel location, based on input pixel value
  - $$g(i,j) = h(f(i,j))$$

  - #### Gamma correction

- ### Local operator

  - compute output value at each pixel location , based on neighbouring pixels around input pixel
  - ex: sharpening filter

  - #### Linear filtering

    - output pixel value is a weighted sum of neighbouring input pixels
    - $⨂$: cross correlation convolution (no flip kernel against image)
    - $$g(i,j) = h(u,v)⨂f(i,j)$$
    - $$g(i,j) = \sum_{u,v} f(i+u, j+v)h(u,v)$$
    - $*$: convolution (flip kernel against image)
    - $$g(i,j) = h(u,v)*f(i,j)$$
    - $$g(i,j) = \sum_{u,v} f(i-u, j-v)h(u,v)$$
    - properties (only applies to convolution)
      - commutative
        - $$f * h = h * f$$
      - associative
        - $$(f*h1)*h2 = f*(h1*h2)$$
      - distributive over addition
        - $$f * (h1+h2) = (f * h1) + (f * h2)$$
      - multiplication cancels out
        - $$kf * h = f * kh = k(f*h)$$

## Common filters

- Guassian filter
- Blur filter
- Sobel filter

## Efficient filtering

- multiple filters

  - generally more efficient to combine multiple 2d filters and filter image once
  - since filter kernels are generally much smaller than image

- seperable filters

  - generally more efficient to filter with two 1D filters than one 2D filters

## Convolution output size

- valid convolution

  - output image is smaller than input image

- ### Border handling

  - pad with constant value (ex: black border)
  - wrap image
    - better with tiling textures
  - clamp / replicate border value
    - better with photos
  - reflect image

## Signals

- any signal/pattern can be described as a sum of sinusoids
- sinsoids: $$y = Asin(\omega x+\phi)$$
- A = amplitude
- $\omega$ = frequency
- $\phi$ = phase

## Fourier analysis

- any signal can be expressed as a sum of sinusoids

## Fourier transform

- decomposes signals into component frequencies
  - values are complex numbers respresenting amplitude & phase of sinusoids
- for each frequency, magnitude (amplitude) + phase
- magnitude capture holistic "texture" of image
- phase captures mainly the edges

## Frequency filtering

- ### Operations in frequency domain

  - spatial domain operations have equiv. operations in frequency domain
  - convolution in spatial domain = multiplication in frequency domain
  - $$FT[h * f] = FT[h]FT[f]$$
  - inverse: $$FT ^{-1} = FT ^{-1}[h] * FT ^{-1} [f]$$

- ### Bandpass filter

  - removes a range of frequencies from a signal

  - #### Low pass filter

    - keep low spatial frequencies, remove high frequencies

  - #### High pass filter
    - keep high spatial frequencies, remove low frequencies

## Applications

- ### Image compression
  - frequency domain is a convenient space for image compression
  - human visual system not sensitive to contrast in high spatial frequencies
  - discard info in high spatial frequencies doesnt change the 'look' of the image

# Week 3

## Image formation

## Diffuse (Lambertian) reflectance

- $$I_{D}(x) = I_{L}RN(x) \cdot L$$
- $I_{D}$ = Intensity of reflected light
- $I_{L}$ = Intensity of light source
- $R$ = Reflectance (color)
- $N$ = Surface normal vector
- $\cdot$ = vector to light source
- ### goal of vision
  - recover surface color & normal from reflected light

## Visible light

- ### spectral power distribution (SPD)
  - relative amount of each wavelength reflected by a surface
- perceived color
  - human color perception is based on 3 types of color sensitive cells
  - cameras also have 3 color sensors, each with different spectral sensitivity
  - most surface reflect range of wavelengths, but perceived color is a function of cone response
  - result: many different spectra appear as the same color
- trichromatic color response
  - sensor reponse = sensitivity $*$ spectrum, integrated over all wavelengths

## Color representation

- RGB
  - most common for digital images
- HSL/HSV (hue, saturation, lightness/value)
  - attempt to match human understanding of color
- CIE 1931 XYZ
  - basedo n human cone sensitivity
- LAB (luminance, a*=red/green, b*=blue/yellow)
  - approx. perceptually uniform space
- color transform
  - convert between dcolor space is trivial

## Shading and surfaces

## Recovering surface normal

- assume no change in surface color/reflectance
- can recover angle between surface normal & light source, but not normal
- $$I_{D}(x) = N(x) \cdot L = \cos \theta_{x}$$
- add. assumptions:
  - normals along boundary of objects are known
  - neighbouring normal are similar
- ### shape from shading
  - recover 3D shapes from 2d images based only on surface brighness (shading)
  - requires add. assumptions

## Recovering surface reflectance

- luminance = reflectance $*$ illumination

## Recovering surface properties

- simple approach: assume lightning is blurry/smooth and hard edges are always due to reflectance

## Cast shadows

- change in illumination, not change in surface

# Edge

- change in intensity (light)

## Cause of edges

- caused by:
  - surface normal discotinuity (diff orientation)
  - depth discotinuity
  - surface color discotinuity
  - illumination discotinuity (shadow)

## Gradient

- math equation
- gradient at a single x, y point is a vector
  - direction is the direction of maximum slope
  - length is magnitude (steepness) of slope

## Issue: noise

- derivative of a row/col of image can be noisy
- solution 1: smooth (blur) image first (convolve with a gaussian blur) before derivative
- solution 2: filter image with derivativ applied to gaussian filter (easier to compute)

## Canny edge detection

- detect edges based on image gradient, + additional processing to improve edge map
- steps:

  - filter with derivative of gaussian filters
  - get magnitude, orientation of all edges
  - (only need 2 oriented filters, dx, dy)

- ### non-maximum supression
  - if nearby pixels claim to be part of same edge, only keep the one with maximum gradient
  - steps:
    - bin edges by orientation (vertical, horizontal, diagonals)
    - for each edge pixel:
      - check 2 neighbor pixels orthogonal to the edge pixel
      - if either has same edge orientation AND higher magnitude, this pixel is not an edge

## Getting edges in low contrast image

- edges dont always have high contrast in image
- ### Thresholding with hysteresis
  - 2 threshold T1, T2 with T1 > T2
  - stong edges = magnitude > T1
  - weak edges = T1 > magnitude > T2
  - for each weak edge:
    - check 8 neighboring pixels
    - if any is strong edge, relabel weak edge pixel as strong edge
  - final edgemap = strong edges

# Edges for image recognition

## Why edges?

- compression
  - edge = discontinuity
  - efficient way to represent images: only represent points where signal changes
- invariance
  - edge based feature are invariant/tolerant to many irrelevant image changes (not sensitive to change)

## Definitions

- ### Invariant
  - invariant to x = response does not vary with x, insensitive to change in x
- ### Tolerant
  - tolerant to x = response is mostly insensitive to x

## Edges invariant to light intensity?

- image derivative is invariant to intensity shift
- tolerant to contrast change, but depends on threshold

- edges tolerant to light direction
- edges invariant to translation
- edges not invariant to rotation
- edges not invariant to scale
- edges somewhat tolerant to 3d rotation/pose

## Image recognition

- recognize objects across variations in lightning, position, size, pose etc
  - learn invariant features & compare them to image
  - learn seperate set of features for each variation & compare each one to image
- recognition algorithm often use mixture of both strategies

# Week 4

## Recognition

- category-level = group level
  - groups can be more/less specific (recognize its a dog)
  - different from instance-level recognition (like what type of dog)
- whole image = 1 label per image
  - diff from detection = locate objects in image
  - diff from segmentation = label individual pixels
- ### Difficulty
  - inter-category similarity
  - intra-category variability
    - diff instances, illumination, scale, viewpoint/posse, background/occlusion
- ### Goal
  - build representation of image that:
    - distinguish diff categories
    - invariant/tolerant to variation within category

## Deep learning "revolution"

- GPUs, fast parallel computation
- algorithm improvements
- internet

## ImageNet

- class images collected online, manually cleaned by humans
- more than 5k classes, commonly used dataset ~1k classes

## Image recognition

- supervised learning problem: map images to classes

## Neural network

- multiple layers of neurons working in parallel on an input
- each neuron on layer L receive input from neurons on layer L-1 & produces 1 output
- neuron's output is weighted sum of inputs, followed by a non-linear activation function
- trains through backpropagation
- compute gradiant of loss function w.r.t network parameters, starting with output layer & propagation to earlier layers, adjusting weights to reduce loss
- learning rate is a free parameter
- loss function based on diff in ground truth & prediction
- ### adv
  - universal approximator
    - able to approx. any continuous function on $R^{n}$
  - feature embedding
    - learn complex features
  - parallelisable
    - within each layer, neurons are independent
- ### disadv
  - large number of parameters
  - high memory/time/data requirements
  - prone to overfitting

## Non-linear activation function

- sigmoid ($\sigma$)
- hyperbolic tan (tanh)
- rectified linear unit (ReLU)

## Convolutional neural network

- why convolution instead of just regular neural networks?
  - more efficient learning of local, repeated patterns
  - disadv: limits what the network can learn

## Convolutional layer

- defined by:
  - a kernel
    - matrix overlaid on image to compute elementwise product with image pixels
  - a stride
    - how many positions in image to advance kernel on each iteration (1 = kernel operate on every pixel of image)

## fully connected layer vs convolutional layer

- fully:
  - each neuron connected to every neuron in input
  - neuron learns some combination of input
  - output to next layer is neuron's response
- colvolutional:
  - each neuron connected to small patch of input
  - neuron learns a convolutional kernel on input
  - output to next layer is input convolved with neuron's kernel

## Convolution output size

- valid convolution (kernel > 1x1) results in output smaller than input
- pad input if same-size output is needed

## Convolution layers

- adv:
  - efficient
    - learn to recognize same features anywhere from image, with fewer parameters compared to fully connected layer
  - preserve spatial relations
    - output is image with values indicating where features are present
- disadv:
  - limited kernel size means model is limited to learning local features

## Downsampling in CNNs

- common to downsample convolution layer output
- reduce output size & number of computations needed in subsequent layers, more efficient for later layers
- improve tolerance to translation

  - small changes in input wont change downsampled output

- ### Strided convolution

  - convolutional stride = distance between successive convolution windows
  - assume no padding:
    - output_size = ceil ((input_size - kernel_size + 1) / stride)
  - with padding:
    - output_size = ceil(input_size/stride)
  - adv:
    - efficient: higher stride, fewer convolution operations
  - disadv:
    - kernel window skips over parts of image, important image features could be missed

- ### Max pooling

  - after convolution, each activation map is seperately downsampled
  - max pool stride determines amount of downsampling
    - output_sie = input_size/stride
  - within a given window in the activation map, take hghest value and discard the rest

- ### Average pooling

  - within a given window in the activation map, average the values

- adv:
  - max pooling likely to preserve most important features, compared to strided convolution / average pooling
- disadv:
  - average pooling 'blurs' over features, imprtant features may be lost
  - pooling slower than strided convolution

## Regularisation in CNNs

- CNN prone to overfitting due to high number of parameters
- common options:

  - ### L1 / L2 regularisation

    - add an additional term to loss function to encourage smaller values for network parameters
    - L1 add $\sum{i}|\theta{i}|$
      - penalise sum of abs value of all parameters
    - encourage sparse representations - many parameters should be 0
    - L2 add $\sum{i}\theta{i}^{2}$
      - penalise sum of squares of all values
      - encourage small (but not 0) parameters
    - free parameters when adding regularisation:
      - how much weight to give regularisation term vs other terms in loss function
      - which layers to include regularisation?
      - which parameters to include?
    - adding regularisation slows down training
    - too much regularision can result in underfitting

  - ### dropout

    - randomly discoard some neurons (some output = 0)
    - force neurons to find useful features independent of each other
    - trains multiple architectures in parallel
    - percentage of neurons to drop is a free parameter
    - can be applied to all or some layers
      - typically later layers have more dropout
      - dropout only used in training
        - evaluating network on new data (validation/test), all neurons are active

  - ### early stopping
    - stop trianing when shown signs of overfitting
    - monitor performance on validation set
      - subset of data not seen in training and not included in test set
      - periodically check model performance on vlaidation set during training - a decrease suggests overfitting
      - encourages smaller values for network parameters, keeping them close to initial values (around 0)

## Training image recognition in CNNs

- ### overview

  - typical architecture for image recognition:
    - some convol layers, with downsampling
    - 1 or more fully conected layers
    - softmax output with cross entropy loss
  - idea:
    - do feature embedding in convol layers
    - fully connected layres effectively a linear classifier to predict class from high level features

- ### Loss function: softmax

  - produces a vector that has properties of a prob. dist.

- ### Loss function: cross-entropy loss

  - measure diff between model & ground truth prob. dist.

- ### Training process

  - split data into train/validation/test sets
  - split training data into batches
  - for N:
    - preprocess batch of image data
    - classify batch, compute loss
    - update model parameters with backprop
  - periodicaly check trained model performance on validation set (for early stopping)

- ### Data preprocessing

  - image whitening: scale image to 0-255 range, then normalise so each pixel has mean=0 and (optional) std=1

- ### Data augmentation

  - manipulate trainng data to generate more samples
  - even smaller networks will overfit without data augmentation
  - common options:
    - random crop
    - horizontal reflection
    - small color/contrast adjustments
  - less common:
    - random rotation (slow)
    - random scale (slow)
    - random occluders

- ### Training process

  - initialise network weights and bias
  - set training parameters
  - monitor training and validation loss
    - Stop training when validation loss no longer decreases

- ### batch size

  - portion of training data used to compute gradient for parameter update
  - not computationally feasible to use whole dataset to compute each update
  - dataset randomly split into N batches of size b
  - N updates = 1 epoch (every iamge seen once)
  - smaller batch size
    - more updates (faster to compute)
    - noisier updates (high variance in gradient)
  - larger batch size
    - fewer updates (longer to compute)
    - more stable updates
  - normally limited by memory constraints

- ### optimiser
  - stochastic gradient descent SGD
  - root mean square propagation Rmsprop
  - adaptive moment estimation Adam
    - keep a moving avg of squared gradient/gradient to divide learning rate
    - diff from SGD that maintains a single learning rate for diff gradients with diff magnitudes
- ### learning rate

  - how much ot change network parameter on each update
  - too high: unstable training
  - too low : very slow learning

- train until model performance on validation set stops improving

# Week 5

## AlexNet innovations

- ReLU (Rectified Linear Unit) activation function
  - faster training
- train on GPU
  - parallelisation allows faster training
- oveerlapping max pooling regions, response normalisation after ReLU
  - small accuracy increase
- data augmentation
  - reduce overfitting
- dropout
  - reduce overfitting

## VGG-16

- stacked convolutional layers
- stack multiple 3x3 convolutional kernels to effectively make larger kernels
  - 2 3x3 conv layers = effective receptive field of 5x5
  - 3 3x3 conv layers = effective receptive field of 7x7
- dont use response normalisation

## GoogLeNet

- ### Inception module
  - choose right kernel size in CNNs is difficult
    - obj/feature can appear at any scale
    - sol: use multi kernel sizes and concatenate
  - add 1x1 convol layers to reduce no of channels (dimensionality reduction)
  - learns features at variety of kernel size/scale
- ### aux classifier
  - used during training only
    - classify images based on early layer representations & update parameters
    - helps with vanishing gradient problem

## ResNet

- performance saturates then decreases with deeper neural networks
- not due to overfitting
  - performance is worse on training set
- should be possible to learn parameters in deep network that would let it act like small network
  - small conv layers learn identity kernels, some learn shallow network's kernels
- deep CNNs cant learn this solution within reasonable training time
- sol: add 'shortcut connections that skip some layers

- ### Residual learning
  - reformulate learning problem
    - training network: input $x$, output $H(x)$, which is feature representation of $x$
    - residual network, input $x$, output $H(x)-x$, which is then added to $x$ to get $H(x)$
  - easier to learn identity mapping
  - #### Residual block
    - simplifies learning problem by making it easier for networks to learn identity mapping
    - allows deeper networks to improve accuracy

## MobileNets

- lightweight architecture for mobile apps
- seperable filters
  - ex: filtering with 2 1D filters = filtering with 1 2D filter
- use depthwise-seperable filters
  - 2D filters in x,y and 1D filter over channels
- depthwise seperable convolution
  - less parameters / computation
  - limits what kernels can learn
  - not all kernels are seperable
- smaller & faster than other architectures
  - lower accuracy
  - better suited for real time app, phones

## Classification results

## ImageNet classification

- 1k obj class
- output = prob dist. over 1k class labels
- top-1 accuracy
  - for each test image, model correct if
    most likely class == ground truth class
- top-5 accuracy
  - for each test image, model is correct if
    any of 5 most likely class == ground truth class

## Generalisation

- features from neural networks are good representations for a range of tasks

## Image recognition: pixels

- pixels are poor space for classification
  - high dimensional space: $256*256*3$ image = 196k attributes
  - irrelevant transformations cause large changes in pixel values

## Image recognition features

- a good feature space for image recognition
  - lower dimensional
    - ex: 1k values per image
    - projects images from same class into similar part of space
    - (image with same clas label have similar features)
- use pretrained networks

  - CNNS convert images from pixels to high lvl features that are good for classification
  - give good performance on a range of compvis tasks
  - ### transfer learning

    - use features from CNN trained on large scale task, with minimal retraining
    - embedding of an put = network's resposne to input at some layer
    - extract rrepresentation from a late layer of a CNN trained on ImageNet
    - use neuron's activation as attribute for chosen classifier
    - more efficient method:

      - remove output layer of CNN trained on ImageNet
      - replace with appropriate output layer for chosen task
      - initialise new layer & train only this layer, freeze all other network parameters
      - or, train some later layers but freeze earlier layers

    - #### Retraning layers

      - finetuning
        - retraining layers of pretrained CNN
      - num of layers to finetune depends on dataset size & its similarity to ImageNet
        - more dissimilar datasets need more retraining of lower layers
        - if dataset size is limited, training lower layers may lead to overfitting

## ImageNet results

- ### What affects performance?
  - larger objects easier to recognize
    - maybe because background is consistent for large objects
  - natural objects easier to recognize than man-made
  - more highly textured objects are easier to recognize

## Model visualization

- ### Visualizing features
  - high dimensional space, cant just plot all images in this space
    - can use dimensionality reduction
    - or look at local regions (what images are near neigbours in this space)
- ### Feature space visualization
  - options for dimensionality reduction
  - PCA (principal component analysis)
    - show dimensions with the most variance
    - simple but often hard to interpret
      - only a few dimensions can be visualized simultaneously
  - t-SNE (t-distributed stochastic neighbor embedding)
    - flattern high dimensional data into 2D/3D
      - near neighbours stay nearby
- ### Visualizing convolutional kernels
  - visualizing 1st convol layer kernels easy
    - input channels are RGB
  - higher layer conv kernels harder
    - channels are high dimensional
    - represent complex features
  - #### Maximally activating patches
    - run many images through the network and find patches that give highest response in this channel
  - #### Guided backprop
    - compute gradient of neuron value w.r.t image pixels
    - which pixels matter most for correct classification
    - does not pass back negative gradient in backpropagation
- ### Visualizing image regions

  - what parts of image most important to deterine class label
    - help show what features the model uses to determine class
    - help debug problems
  - occlusion method: mask image and see how much class probability changes
  - #### CAM (class activation mapping)

    - add global average pooling (GAP) layer before classification layer
    - use weights of this layer to determine where the class-relevant features are
    - disadv:
      - GAP layer must be added to pretrained network and then finetuned
      - Only allows visualisation of the last layer

  - #### Grad-CAM (gradient-weighted class activation mapping)
    - take response from some layers
    - compute gradient of class score w.r.t layer reponse
    - global average pool (avg over image x,y) the gradients to get a vector of weights (1 weight per channel)
    - compute activation map ReLU

- ### Visualizing classes
  - usually based on gradient ascent
    - synthesize image that maximises class label response
      - initialize image with 0s or small random noise
      - run image through network, compute gradient
      - update image pixels to minise loss
    - problem: many possible arrays of pixels can generage very high model response
      - not all will look like real images

## Invariance / tolerance

- are features learned by CNNs invariant?
- ### Generalisation
  - models very sensitive to some types of noise
  - performance of top models on ImageNet vs ImageNetV2
  - drop of ~10% performance suggests some overfitting to quirks of ImageNet
- tolerant to variation included in training data
- not tolerant to variation that didnt appear in training data
- classification rely on recognizing few key features / local texture elements

# Week 6

## Approaches to recognition

- detect local features, ignore spatial position
  - ex: bag of words / features
- local features + weak spatial relations
  - spatial pyramid models
- detect local features & model spatial relations between them
  - deformable-parts models
  - keypoint tracking / matching

## Recognition from local features

- ### bag of words
- ### bag of features
  - based on methods from NLP
    - represent a document using historgram of word frequency
    - in an image, "words" are local features
  - words -> features
    - problem: in image, same "word" can have many appearances
    - solution: combine similar local features
      - ex: with k-means clustering
  - ### spatial pyramids
    - idea: run bag of features at multiple scales
    - difference:
      - detecting features at 1 scale & pooling at multiple scales
      - detecting features at multiple scales
- works well for category-level recognition
  - high invariance to object translation and pose

## Feature detection

- ### Dense vs sparse features
  - dense feature representation: compute local features everywhere
  - sparse feature representation: compute local features only at a few "important" points
- feature detection
  - finding "important"(interest point / keypoints) points in image
  - points that can be detected reliably across image transformations
- feature descriptor: a short code / set of numbers to represent a point in image
- ### Selecting good keypoints
  - should be easy to recognize in a small window
  - shifting the windiw in any direction should produce a large change in intensity

## Corner detection

- change in appearance of window w(x,y) for the shift [u,v]
  - $$E(u,v) = \sum_{x,y}w(x,y)[I(x+u, y+v)-I(x,y)]^2$$
  - $w(x,y)$ = window function
  - $I(x+u, y+v)$ = shifted intensity
  - $I(x,y)$ = intensity
- common window functions: square, gaussian
- ### maths
  - approx. shifted intensity using Taylor series

## Corner response function

- detect corners using eigenvalues
- lock for points where $\lambda_{1}\lambda_{2}$ is high, and $\lambda_{1}+\lambda_{2}$ is low

## Harris corners

- $\lambda_{1}\lambda_{2}$ and $\lambda_{1}+\lambda_{2}$ are the determinant and trace of matrix M
  - det = np.linalg.det(m)
    - det(M) = $\lambda_{1}\lambda_{2}$
  - trace = m.trace()
    - tr(M) = $\lambda_{1}+\lambda_{2}$
- harris corner response
  - $R = det(M) - k(tr(M))^2$
    - $k$ determined empirically, around 0.04-0.06
- ### alternatives
  - alt corner response functions
    - shi-tomasi
    - brown, szeliski & winder
  - alt corner detectors
    - blob detectors
    - ML-based detectors

## Invariance / tolerance

- corner detection based on image gradient (edge)
  - invariant to translation
  - tolerant to changes in lighting
- because corner response is based on eigenvalues, it is invariant to image-plane rotation
- not invariant to scale

## Feature descriptors

- need a way to represent keypoints found in an image

## Scale-invariant feature transform (SIFT)

- compute gradient, take histograms in a grid of pixels around interest point
- weight gradient magnitudes based on distance from centre of patch
- normalise histograms to sum to 1
- implementation details:
  - patch size = 16 x 16 pixels
  - grid = 4 x 4 cells
  - histogram bins = 8 orientations
  - gaussian weighting from central of patch
  - 2 step normalization
    - normalise to sum to 1
    - truncate values to 0.2
    - normalise to sum to 1
- descriptor length = 4 x 4 x 8 = 128

## Feature matching as model fitting

- not just looking for same features, but same spatial arrangement of features
- model fitting problem:
- propose a model that explains correspondence between matched points
- find points that fit model / measure goodness of fit / find model parameters
- problems:
  - outliers (data not explained by model)
  - noise in data (explained by model)

## Hough transform

- each edge point 'votes' for the lines that pass through the point
- identify lines with most 'votes'
- identify parameters with most votes is easy if no nosise in point locations
- usually there is some noise
- solution to noise: bin parameters
- points vote for bins
- another problem: slope & intercept (m,b) are unbounded
- solution: use polar representation of (m,b)
- parameters:
  - bin size
  - threshold for peaks
  - no of peaks (= no of lines)
  - min line length
  - max allowed gap (to treat segments as same line)
- basically grid search over all possible values of each parameter, so limited to just a few parameters

## RANSAC

- RANdom Sample Consensus
- like hough transform, points 'vote' for model that explains those points
- iteratively:
  - sample N points at random (N = no of points needed to fit model)
  - calc model parameters
  - count no of points explained by model (inlier points)
- best model = model that explains most points
- no of iterations
  - should be high enough that it is very likely (prob=0.99) to obtain at least 1 sample with no outliers
- threshold for inliers
  - choose $\delta$ so that a good points with noise is likely (prob=0.95) within threshold

## Feature matching

- exhaustive matching is slow (O(mn)), m,n = no of keypoints in each image
- more efficient option: approx. nearest neighbours
  - faster, may not find closest match
- even with ratio matching, proportion of false matches is likely to be very high

## Affine transformations

- any combination of translation, scale, rotation, shear
- lines map to lines
- parallel lines remain parallel
- ratios preserved

## Projective transformation

- combine affine transformation with a projective warp
- lines map to lines
- parallel lines **not** parallel
- ratios **not** preserved

## Week 7

## Cues for depth

- 2D image contain variety of info for depth perception
- cues available in a single view include perspective texture & object cues
- more accurate depth info can be obtained by combining multiple views (stereo, motion)

## Depth from stereo

- stereo pair
  - image from 2 cameras with horizontal shift in camera position
- assume:
  - image planes of cameras are parallel to each other and to the baseline B
  - camera centres are at the same height
  - focal lengths f are the same
- goal: find z
- $$z=\frac{fB}{x-x'}$$
- $f$ = focal length
- $B$ = baseline
- distance z is inverly proportional to disparity $x-x'$

## Basic stereo matching algorithm

- for each pixel $x$ in image:
  - scan a horizontal line in other image, find best match $x'$
  - compute disparity $x-x'$ and compute depth
- SSD - sum of square difference
  - $$SSD = \sum_{i,j}(w_{i,j}-w'_{i.j})^2$$
- normalize cross correlation
  - $$norm.corr. = \frac{\sum_{i,j}w_{i.j}w'_{i.j}}{||w_{i.j}||||w_{i.j}||}$$
- example: autostereogram

## Effect of window size

- smaller window
  - finer detail, more noise
- larger window
  - smoother depth, lack detail

## additional constraint

- individual matches are often ambiguous
- set of matches should obey additional constraints:
  - uniqueness
    - point in 1 view has no more than 1 match in other view
  - ordering
    - corresponding points should be in same order in both views
    - does not always hold
  - smoothness
    - disparity values should change smoothly (for the most part)

## applying constraints

- $$E(D) = \sum{i}(W_{1}-W_{2}(i+D(i)))^2 + \lambda\sum_{neighbours i,j}\rho(D(i)-D(j))$$
- minimize E(D) using optimisation method like graph cuts

## Rectification

- used when image planes are not parallel
- find projective transform that maps each image to same plane

## Single-view depth

- ### Supervised depth classification
  - treat depth estimation as classification task
    - for each pixel in image, predict distance from camera
  - train on images with annotated depth maps
  - #### loss function
    - images may have v. large range of depths
      - a loss based on log(depth) may work better than abs. depth
    - mean depth of scenes can vary widely
    - to discourage models from simply learning mean depth, scale the loss function so it is similar for diff scenes
- adv:
  - dont require multiple views / steroe camera
- disadv:
  - 'blurry' depth at object edges (can be combined with edge maps for better result)
  - models may not generalise well to new contexts
- 2d image provide multiple cues for 3d depth
- more accurate depth measurements from multiple views (stereo) than on single image

## Depth from disparity

- instead of training on annotated deph maps, train on stereo image pairs
- adv:
  - stereo image pairs can be produced with standard cameras, whiledepth maps require special equipment like LiDAR
- step:
  - input: 1 image from a stero pair (ex: left)
  - learn to predict disparity map that will produce the other image (ex: right)
  - distance from camera to surfaces can be computed from disparity map
  - train on stereo pairs
- loss is sum of:
  - appearance loss (diff between original & predicted images)
  - disparity smoothness loss
  - left-right consistency loss (diff between disparity maps)

## Multi-view problem

- solve for:
  - camera motion
    - what is the transform that relates the 2 views
  - camera parameters
    - ex: focal length
  - scene geometry
    - given corresponding image points in 2 views, what is position of point X in 3D space

## Camera calibration

- ### Camera parameters
  - Intrinsic parameters
    - camera parameters related to image formation
      - ex: focal length, optical centre, lens distortion
  - Extrinsic parameters
    - camera pose (location & orientation) relative to the world
- camera calibration is a process to find intrinsic parameters
- these parameters learned from image data with unknown extrinsic parameters
- ### calibration method
  - requires a calibration target, a planar surface with known pattern that is easily detected/tracked by feature detection methods
    - common choices: checkerboard, squares, circles
  - take multiple photos / video of calibration target in many diff poses
  - solve for intrinsic & extrinsic parameters
- ### calibration algorithm
  - given multi image, solve for H, & camera matrix using a system of lin. eq.
  - this model assumes no lens distortion
  - given best fit for H, estimate distortion parameters
    - diff formula for diff distortion models
  - iterate to refine parameters
- ### calibration result
  - output of calibration process is an estiamte of camera intrinsic parameters
    - camera matrix, lens distortion parameters
  - allows for accurate mapping between image coords & word coords
- ### alternative methods
  - calibration using planar surfaces in the world
    - adv: no need for special calibration target
    - disadv: more diff to detect/track keypoints, may introduce errors
  - look up camera parameters from manufacturer specs
    - adv: no computation
    - disadv: only for cameras with fixed focal length

## Homogeneuos coordinates

- convenient to use homogeneuos coordinates when converting between world & image points
- image points represented with 3 vals (x,y,z)
- 3rd value can be thought as distance to image plane

## Projection model

- pinhole projection model can be represented as matrix in homogenous coordinates

## Epipolar geometry

- ### limitations
  - system only solved up to a scaling factor, need at least 1 known distance to solve for real world positions
  - degenerate case: cant solve if system as too few degrees of freedom
    - points in world are all coplanar
    - camera translation = 0 (just rotation)
- epipolar geometry describes relations between points in 2 views
- a point in 1 image lies along an epipolar line in the other image
- epipolar lines in an image meet at a point called the epipole
- epipole is the projection of 1 camera in the other image

## two-view problem

- find camera transform (translation + rotation) that relates 2 views

## Week 8

## Texture

- region with spatial stationarity
  - same statistical properties everywhere in the region
- texture is a 2D surface applied to a 3D model
- ### types of textures
  - periodic texture
    - has subregion that repeats in a regular pattern
  - stochastic (aperiodic) texture
    - generated by a random process
- ### texture models
  - parametric models
    - represent texture with a set of adjustable parameters
  - non-parametric (stitching) models
    - represent texture as image patches
  - used in texture synthesis
    - create more of a texture
    - computer graphics, video games
    - image inpainting
  - used in texture transfer
    - artistic effects
    - online shopping

## Texture synthesis

- ### image stitching
  - non-parametric texture sysnthesis
    - randomly sample small patch from original image
    - spiral outward, fill missing pixels by finding similar neighborhoods in original texture
    - (neighbourhood size is a free parameters that specifies how stochastic the texture is)
- ### Image quilting
  - efficient patch-based texture synthesis
  - use existing patches of texture to synthesis more texture
  - main problem is connecting them tgt without visible artefacts/seams
  - step:
    - choose a patch with overlap size
    - initialzie with random patch
    - for each subsequent patch
      - find a patch in original texture that is most similar to region, considering only pixels in overlap region
      - seamlessly paaste in patch by cutting a path with minimum overlap error
  - #### graph cuts
    - represnet neighbouring pixels as graph
    - edge weight = overlap error
    - problem
      - find path through graph with minimum total overlap error
- ### Image inpainting
  - similar idea to fill in missing regions of an image
    - find similar patch in another image
    - paste in patch with an error-minimizing cut
- ### parametric texture synthesis
  - alternative to stitching approach
    - repesnet texture with no of parameters
  - to synthesize texture, coerce a noise image to match required parameters (gradient descent)
  - #### fourier texture synthesis
    - synthesize texture by matching fourier magnitude
    - ok results for simple texture, dont work well in general
  - texture could be defiend as distribution over simple features, like color / edge orientation at various scales
  - synthesize texture by matching distribution
  - simple distributions of features not enough
  - need to represent feature co-occurence
  - set of statistics needed to represent real images may be very complex
  - instead of modelling statisics by hand, represnet texture as feature response in layers of a neural network trained on ImageNet classification

## feature correlations

- texture represented as correlations between feature map at a layer of neural network

- non parametric texture synthesis based on copying texture patches
  - adv: work well on periodic textures
  - disadv: no model of texture parameters
- parametrix texture synthesis represents textures in terms of set of parameters
  - adv: most methods work better on stochastic textures
  - disadv: even very complex model may be incomplete

## Texture transfer

- render an image in style of another image
- ### neural style transfer algorithm
  - both images (content, style) are run through a VGG network trained on ImageNet
  - content is represented as response from a layer of neural network
  - style is represented as correlations between feature maps at a layer of neural nework
  - use gradient descent to find an image taht matches both content & style
  - #### style transfer parameters
    - loss is sum of loss from content reconstruction & style reconstruction
    - relative weight of content vs style is a free parameter
    - content & style can be matched at any combination of layers
    - generally, match content at higher layers, style across all layers

## Shapes

- models of 2d shapes usually based on:
  - bounding contour of shape (segments, angles)
  - internal structure of shape (branches)
- ### shape skeletons

  - topological skeleton
    - thinnest possible version of a shape
  - formed of lines that are equidistant from boundaries of the shape
  - geometrical description
    - skeleton points are centrepoints of largest discs that can be fit inside the shape
    - if the shape was painted with circular brush, skeleton would be path of brush
  - #### skeletonisation algorithm
    - grassfire transform
      - algorithm for shrinking/thinning a shape
    - for each pixel in shape, compute distance to closest boundary; peaks in distance map are the skeleton
  - #### skeleton representation
    - skeleton + distance to boundary at each skeleton pixel is compact, invertible representation of shape
    - to 'inflate' skeleton, place a disk at each skepeton pixel
  - #### application:
    - ##### shape recognition
      - shape skeletons easily converted to graphs
      - graph representation can be sued for shape matching, pose recognition
    - ##### 2D -> 3D
      - shape skeletons can be used as basis for simple 3D model
        - just 'inflate' with spheres instead of disks
  - #### drawbacks
    - shape must be segmented from background
    - small change in shape boundary procude large change in skeleton

- ### Contour representation
  - #### active contours (snakes)
    - parametric model that fits itself to object boundary
    - "shrink wraps" around object to capture shape
    - ##### algorithm
      - initialize countour outsdie object boundary
      - on each step, allow each point on the contour to shift 1 pixel in any direction
        - shift to minimise a loss function
        - $$E_{total}=\alpha E_{elasticity} + \beta E_{stiffness} + E_{edge}$$
      - repeat until loss doesnt change
    - ##### application
      - ###### segmentation
        - active contours used for segmentation & tracking, particularly in medical image analysis
    - ##### drawbacks
      - require initialisation (often from human annotator)
      - may not fit shape correctly
        - tradeoff between elasticity/smoothness & edge-matching
          - may fail to fit concavities in complex shapes
        - difficult to detect shapes in clutter
- ### Face models
  - difficult to develop a general-purpose model of shape that can represnet all possible shapes well
  - however, possible to develop parametric mode lfor particular classes of shape
  - 1 widely-studeied class of shapes is human face
  - #### eigenfaces
    - if faces arealigned, pixel luminance values are sufficient to capture face shape
    - simple pixel-based model
    - ##### algorithm
      - each face represented as vector to the mean face image
      - parameters of face shape are obtained from PCA of face vectors
    - problem: usually cant assume face appear in consistent alignment/lighting
    - need models that can consider shape/pose in real world conditions
  - #### active appearance models
    - label corresponding landmark points in each image
    - warp images onto the mean shape to get shape-free texture
    - obtain "shape", "texture", & "appearance" (shape+texture) parameters through PCA
    - to fit model to new face, use gradient descent to minimize difference between model & image
    - application: face synthesis, face segmentation
    - active appearance models seperate shape & texture
      - allow alignment of facial features, even when images are not aligned
    - problem: shape is represented using 2D contours
      - cant seperate face shape vs pose
      - cant seperate surface color vs lightning
    - ##### 3D face models
      - 3D version of active appearance model: morphable 3D mesh + texture map
      - parameters based on PCa of large 3D dataset
    - ##### application
      - ###### facial recognition
        - most recognition algorithms use shape model to align faces as first step
        - once flaces are aligned, standard CNN pipeline can be trained for face recognition

# Week 9

## Image Generation

## Generative models

- ### Discriminative models
  - learn conditional prob. of class Y given attributes x, $P(Y|X)$
  - input is an image
  - output is a prob. density function over labels $P(Y|X)$
- ### Generative models
  - learn joint prob. of attributes X and class Y, $P(X,Y)$
  - generate new samples from learned distribution
  - #### (Conditional) generative model
    - input is a label
    - output is a prob. density function over images $P(X|Y)$
  - #### (Unconditional) generative model
    - output is a prob. dist. $P(X)$
- Generative models contains discriminative models
  - can use joint prob. to get $P(Y|X)$
  - AND generative can do the reverse $P(X|Y)$

## Bayes' rule

- $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$

## Autoencoder

- ### Unsupervised learning
  - learn a model for unlabelled data
  - goal is to find a model that represents the data as well as possible, usually with fewer parameters
  - uses:
    - simpler model for another ML system
    - form of dimensionality reduction
    - potentially better generalization
- ### Unsupervised learning + NNs
  - like supervised ML algos, unsupervised algos may not work well in 'raw' input spaces (text, images)
  - embeddings (NNs) might work better
    - but we have no labels to learn the embeddings for our task
    - embeddings learned for a diff task may not give a complete representation of the data
- essentially NNs for unsupervised learning
- sometimes called 'self-supervised' learning
- output of network is w/e passed into the network (ex: image)
- hidden layer learns lower dimensional representation of input
- ### Encoder/decoder architecture
  - encode in a hidden layer
  - hidden layer is smaller than input
  - decode to an output layer
  - often encoding & decoding weights are forced to be the same
  - goal: output the image
  - hidden layer
    - 'bottleneck' layer - smaller than input
    - represent the input in terms of latent variables
    - in simplest case, this layer learns PCA
- ### output & loss
  - unlike a standard NN, output is not a class or regression value, its the same type as input
  - activation function is chosen appropriately:
    - for a binary image, tanh / sigmoid
    - for a regular image, linear activation
  - loss function = diff between input & output (ex: MSE)
- unsupervised learning (no labels)
- learns latent representation from data
  - lower dimensional set of features that explain the data
  - not a true generative model - no way to sample new data
    - could 'sample' by giving random latent variable values to the decoder, but no gurantee these will produce real images
- ### Variational autoencoder (VAE)
  - probabilistic version of autoencoder
  - learn latent representation & sample from model to generate new images
  - assume images are generated from some dist. over latent variables $z$
  - assume a simple prior $P(z)$, ex: uniform / Gaussian dist.
  - probabilistic decoder learns $P(x|z)$
  - probabilistic decoder learns $P(z|x)$
  - goal: maximize likelihood $P(x)$
  - adv:
    - learn approx of $P(z|x)$ and $P(x|z)$, where z is latent variable representation of input x
    - can be used to generate new instances of x
  - disadv:
    - outputs often blurry
  - #### Probabilistic decoder
    - input: latent variables $z$
    - output: mean $\mu_{x|z}$ and diagonal covariance $\Sigma_{x|z}$, parameters of a Gaussian dist. that generates $x$ conditional on $z$
    - $P(x|z) = N(\mu_{x|z},\Sigma_{x|z})$
    - goal: maximize $P(x) = \frac{P(x|z)P(z)}{P(z|x)}$
  - #### Probabilistic encoder
    - input: image $x$
    - output: mean $\mu_{z|x}$ and diagonal covariance $\Sigma_{z|x}$, parameters of a Gaussian dist. that generates $x$ conditional on $x$
    - learn approximation of P(z|x):
      - $q(z|x) = N(\mu_{z|x},\Sigma_{z|x})$
    - $P(x|z) = N(\mu_{z|x},\Sigma_{z|x})$
  - #### loss function
    - goal: maximize likelihood $P(x)$
    - loss is based on variational lower bound on $P(x)$
    - consist of 2 terms: reconstruction loss & regularisation loss
    - reconstruction loss
      - encourage network to create output images as similar as possible to input images
    - regularisation loss
      - encourage network to learn a latent representation $z$ that is similar to the prior (standard norm. dist.)
    - properties of latent space
      - to be useful for generation, latent sspace should be:
        - continuous: nearby points in latent space correspond to similar images
        - complete: every point in latent space corresponds to a valid image
      - standard norm. dist. satisfies both requirements
      - use of diagonal covariance matrices ensures latent variables are independent
- ### Application: Sampling new images
- ### Application: Image manipulation
  - given latent variable representation z of image x, can change values of z to create variations on x
  - nearby points in latent space correspond to similar images (continuity requirement) & axes are independent
  - but directions in latent space may not correspond to recognizable image properties (without add. constraints)

## GAN architecture

- Generative Adversarial Networks
  - neural networks that learn to generate instances from a particular dist. (ex: images of faces)
- consist of 2 neural networks
  - generator
  - discriminator
- training involves competition between the 2 networks
- adv:
  - generate samples from complex prob. dist. wihtout actually representing the dist.
- disadv:
  - can be unstable / hard to train
  - diff to evaluate
  - even models that dont show complete mode collapse tend to have lower-than-desired diversity
- ### Generator
  - dont actualy learn the prob. dist., but learns to sample from it
  - input is a latent variable z with a simple prior (ex: uniform random / standard normal)
  - output is an image
  - learn a function to map p(z) to a dist. that approx. p(x)
- ### Discriminator
  - learn to identify real vs fake input created by generator
  - neural network classifier with 2 output classes (real, fake)
  - architecture depend on tasks
    - ex: for images, discriminator might be a CNN with several convolutional layers, followed by softmax
- ### Training
  - networks are trained tgt on a combination of real data X and generator input Z
  - given generator G & discriminator D:
    - D's goal is to correctly classify real vs fake
    - D want to maximize D(X) and minimize D(G(Z))
    - G's goal is to fool the D
    - G want to maximize D(G(Z))
  - can treat as a zero sum game with goal of finding equilibrium between G & D
  - GAN training objective is a minimax game
  - if discriminator too good:
    - easily rejects all fake inputs
    - not much info to train the generator
  - if discriminator too bad:
    - easily confused by fake inputs that dont look real
    - generator will learn a poor solution
  - training can be difficult
    - hard to find balance between discriminator & generator

## GAN Evaluation

- GAN equilibrium dont necessarily mean GAS has found a good solution
- how to tell if a GAN has learned?
  - outputs should not be identical to input (memorised training data)
  - outputs shold look like inputs (looks 'real' and not 'fake')
  - outputs should be as diverse as real data (avoid **mode collapse** = generator only creates 1 or few outputs)
- successful training does not necessarily mean the generator’s output is similar to p(x)
- ### evaluate realism
  - gold standard: human evaluaiton (slow & expensive)
  - auto methods compare responses of an image classifier (ex: CNN trained on ImageNet) to real vs GAN-generated images
  - #### inception score
    - within a class, all images should be confidently classified with the correct label
    - across classes, GAN should produce a wide variety of confidently-classified images
    - adv:
      - automatic, efficient
      - neural network responses correlate with human judgements of image quality
    - disadv:
      - dont require high diversity within categories
      - sensitive to noise, adversarial images
- ### diversity
  - GAN isnt just memorizing training examples
  - does it capture all diversity in training set?
  - #### birthday paradox for GANs
  - suppose a generator can produce N discrete outputs, all equally likely
  - exp: take small sample of s outputs and count duplicates
    - odds of observing duplicates in sample size s can be used to compute N
    - sample of about $\sqrt{N}$ outputs is likely to contain at least 1 pair of duplicates
  - most GANs tested produced about same diversity as was in training set

## Conditional GANs

- conditional model: learn P(X|Y) rather than P(X)
- both discriminator & generator take y as additional input
- CycleGAN
  - train a pair of generator to map x->y and y->x

# Week 10

## Image segmentation

## Pixel clustering

- ### Color clustering
- ### K-means
- ### Gaussian mixture model
- ### Mean shift clustering
  - assume pts are samples from an underlying prob. density function (PDF)
  - compute peaks of PDF from density of pts
  - mean shift algorithm
    - for each pt:
      - a) centre a window on that pt
      - b) compute mean of data in search window
      - c) centre search window at new mean location
      - d) repeat b,c until convergence
    - assign pts that leads to nearby modes to same cluster
    - free param: kernel (commonly Gaussian), bandwidth
  - mean shift segmentation
    - cluster in spatial+color space: ex: (x,y,R,G,B)
- fast simple approach to image segmentation
- mean shift clustering in color + spatial domain
  - auto discover no of clusters; no need to choose k
  - need to choose bandwith
- seperate colors regions
  - (but) regions may not correspond to objects

## Superpixels

- **oversegmentation** methods segment image into regions that are smaller than objects
  - objects seperated from background
  - but objects are also seperated into many parts
- superpixels = groups of adjacent pixels with similar characteristics (ex: color)
- ### SLIC superpixel algorithm
  - initialise cluster centers on non-edge pixels
    - initialise k cluster centres $c_{k} = [x_{k},y_{k},l_{k},a_{k},b_{k}]$ by sampling image in a regular grid
    - for each centre $c_{k}$, check N x N neighborhood around to find pixel with lowest gradient, set $c_{k}$ to this pixel's $[x, y, l, a, b]$
  - for each cluster centre $c_{k}$:
    - in a $2M$ x $2M$ square neighborhood around $c_{k}$, measure pixel dissimilarity to $c_{k}$
    - assign pixels with dissimilarity < threshold to cluster k
    - compute new cluster centre $c_{k}$
  - repeat until avg change in cluster centres (L1 distance) falls from below a threshold
  - dissimilarity measure :
  - $$D = D_{lab} + \frac{\alpha}{M}D_{xy}$$
  - $D_{lab}=\sqrt{(l-l)^2+(a-a_{k})^2+(b-b_{k})^2}$
  - $\alpha$ = weighting parameter
  - $D_{xy} = \sqrt{(x-x_{k})^2+(y-y_{k})^2}$
  - dissimilarity metric dont gurantee clusters will be connected pixels
  - to enforce connectivity, pixels not connected to main cluster are reassigned to closest adjacent cluster
  - #### applications
    - multipurpose intermediate image representation
    - more compact representation for algos with high time complexity (600\*800 pixels -> 200 superpixels)
    - common application: object segmentation
      - oversegment image
      - combine superpixels to find objects
- ### Superpixel merging
  - #### Region Adjacency Graph (RAG)
    - vertices = image regions (pixels / superpixels)
    - edge weights = diff between regions
  - to merge superpixels:
    - identify edges below a threshold & relabel superpixels connected by these edges as 1 region
    - or iteratively:
      - find lowest weight edge, relabel connected superpixels as 1 regions
      - recompute RAG, repeat until a criteria is met (ex: all edges above a threshold)
- ### application
  - intermediate representaiton used as first step for:
    - segmentation (graph-based methods)
    - object detection / localisation
    - video tracking
- compact, intermediate representation used as first step for:
  - segmentation (especially graph based methods)
  - object detection / localization
  - video tracking

## Graph-based segmentation

- images as graphs
  - repesent image as a graph $G=(V,E)$
    - vertices = image regions (pixels / superpixels)
    - edge weights = similarity between regions
- ### graph cuts
  - consider image as fully connected graph
  - partition graph into dispoint sets A, B to maximize total edge weight = remove low weight edges between dissimilar regions
  - minimize value of cut
    - $$cut(A,B)= \sum_{u \subset A,v \subset B} w(u,v)$$
    - w = weight of edge connecting u and v
  - not ideal for image segmentation
    - tends to create small, isolated sets
- ### normalized cuts
  - minimize cut value as fraction of total edge connections in entire graph
  - $$
    Ncut(A,B) = \frac{cut(A,B)}{assoc(A,V)}+\frac{cut(A,B)}{assoc(B,V)} \\
    = \frac{\sum_{u \subset A,v \subset B} w(u,v)}{\sum_{u \subset A,t \subset V} w(u,t)} + \frac{\sum_{u \subset A,v \subset B} w(u,v)}{\sum_{v \subset B,t \subset V} w(v,t)}
    $$
- ### GrabCut
  - segment image pixels into just 2 classes:
    - foreground (object) & background
  - use color clustering + graph cuts to find optimal classification of pixels into each class
  - require user to initialise algo with a bounding box
  - ### algorithm
    - for each class, represent dist. of pixel color as gaussian mixture model (GMM)
    - represent image pixels as graph (8-way connectivity)
    - denote pixel graph as G, and GMM as $\theta$
    - $\alpha$ indicates label of each pixel (foreground / background)
    - iterate until convergence
      - find graph cut to minimize $$E(\alpha, \theta, G) = U(\alpha, \theta, G) + \gamma V(\alpha, G)$$
      - $U(\alpha, \theta, G)$ = -log likelihood of cluster assignment in GMM
      - $\gamma$ = weighting parameter
      - $V(\alpha, G)$ = smoothness penalty based on color similarity, applied to neighbouring pixels iwth diff labels in $\alpha$
    - recompute GMM for new label assignment
- remove edges to break graph into subgraphs, generally try to optimize:
  - similarity within connected region
  - dissimilarity across disconnected regions
  - smoothness / connectivity of connected regions
- normalized cuts - segments into multiple regions
- GrabCut - segment into foreground / background

## Segmentation as classification

- ### Pixel classification
  - image segmentation as classification problem
  - given a window (N x N pixels), classify central pixel
  - classifying individual pixels very slow
  - but CNN can classify multiple pixels in parallel
  - #### parallel patch classification
    - fully connected network (FCN) has only convolutional layers, no fully connected layers
    - last layer is a spatial map
    - can accept any size image as input, output map size depend on input size
    - standard CNN architecture has 2 problems:
      - receptive field size is linear with number of convol layers
      - most methods downsample (ex: with maxpool) to reduce computation
    - solution: use encoder-decoder structure
    - encoder downsamples image, decoder upsamples
    - ##### upsampling
      - max unpooling
        - each upsampling layer is paired with a downsampling layer
        - locations of max items are saved & passed to upsampler
    - ##### transposed convolution
      - convolution with stride > 1 is a form of downsampling
        - ex: stride = 2 means filter moves 2 pixels input for every 1 pixel in output
        - "learnable downsampling"
      - can reverse to do upsampling
        - ex: filter moves 2 pixels in output for every 1 pixel in input
      - **transposed convolution**: convol with a stride < 1
      - can express convol as matrix multiplication
      - tranposed convol mutiplies by transpose of the same matrix
    - ##### 1x1 convol layer
      - convol layer with a 1x1 kernel
      - common in last layer of fully connected network
      - ###### loss function
        - at each pixel, compute cross entropy loss between predicted & known classes
        - $$E=-\frac{1}{N}\sum_{i=1}^{N}y_{i}\log(\hat{y}_{i})$$
- semantic segmentation can treat as pixel classification problem
- can do efficiently with fully convolutional network
- encoder-decoder architecture:
  - dowwnsample with maxpool, strided convol
  - upsample with max unpool, transposed convol
- output is a label for each pixel

- ### U-Net
  - originally for medical image segmentation
  - encoder-decoder structure with some add. features
  - in decoder, upsampled feature map is concatenated with original features from corresponding encoder layer
  - for cells segmentation, U-Net use only 2 classes & weighted cross entropy loss: edges between cells have higher weight
  - also used as encoding-decoding network for:
    - image denoising
    - image inpainting
- ### Instance segmentation
  - semantic segmentation classifies pixels, cant distinguish between instances
  - instead of giving each pixel an instance label, extract patches of image that are likely to be seperate instances
  - do segmentation within each patch
  - commonly used architecture: mask R-CNN
  - #### R-CNN
    - region based convol neural network
    - efficiently extract 'regions of interest' (image patches likely to contain objects) for further processing
  - #### Mask R-CNN
    - take patches extracted by R-CNN & run them through fully convol network
    - FCN predicts a binary segmentation mask ('object' or 'background')
- segmentation can be approached in diff ways
  - clustering / graph based methods
  - pixel classification with FCNs
- adv of FCNs
  - better able to handle complex objects/backgrounds
  - likely to get better results for classes on which they are trained
- disadv of FCNs
  - worse at capturing precise boundary details
  - may not generalsie to classes outside training set

# Week 11

## Object detection

- ### classification vs detection
  - object detection = locate obj in an image
    - classification: is this a \<class label\>?
    - detection: where is the \<class label\>?
  - object detection usually modelled as classification task performed within patches of an image
    - detection: for every patch, is this a \<class label\>?
  - #### sliding window
    - free parameters:
      - stride
      - scale (size of window)
      - shape (aspect ratio)
    - generally obj dimensions are unknown, so a range of scales/shapes is required
    - another parameter: threshold for detection
    - windows over the threshold will be considered 'target'
  - window evaluation: IoU
    - intersection over union between true bounding box and detection window
    - IoU = Area of overlap / Area of union
  - problems:
    - large no of possible windows (slow, increase prob. of false detections)
    - overall evaluation of images with multiple targets can be complicated (multiple targets, multiple detection windows, diff IoUs)

## R-CNN

- sliding window classification
  - very large no of windows per image
- even in a neural network, classifying all possible boxes is slow
- solution: focus on boxes most likely to be objects
- R-CNN: region-based CNN
- given an image, identify small no of windows for object detections (region proposals or region of interests (ROIs))
- use selective search to generate ROIs
  - use superpixels
- selective search algorithm:
  - oversegment images into superpixels
  - iteratively combine adjacent superpixels based on similarity in color, texture, size, compactness
- ### training
  - CNN pretrained on ImageNet
  - last layer (1x1000) is replaced with new classification layer of size 1x(N+1)
    - N+1 = N object classes + 'background' class
    - CNN is retrained on (N+1)-way detection, using regions with IoU >= 0.5 as ground truth 'objects'
    - sample regions so 75% of training set is 'background'
  - CNN features are used as input to:
    - label classification model (1-vs-all linear SVM)
    - bounding box model (class-specific linear regression)
- ### testing
  - input test image
  - compute region proposals (selective search)
  - each region: run through CNN to predict class labels & bounding box transforms
  - 'detections'= regions with highest class confidence scores
- ### advantage
  - much more efficient than classifying every window
- ### disadvantage
  - still requires classifying many windows (ex: 2000)
  - region proposal step could miss some objects

## Fast R-CNN

- major change: run whole image through a fully-CNN
- take region proposals from last convol layer
- use selective search to generate region proposals (like R-CNN)
- backbone network extracts features from image ('feature embedding')
- mapping features to label/bbox is not too complex
- ### training
  - traion on R regions sampled from N images
    - efficency: N is small, R is large
    - sample regions so 75% of training set is 'background'
  - train with multi-task loss
- ### advantage
  - faster than R-CNN (9x faster training, 140x faster testing)
  - slightly more accurate than R-CNN
- ### disadvantage
  - ROIs arent learned
  - region proposal step could miss some objects

## Faster R-CNN

- major change: network learns region proposals, instead of using selective search
- ### region proposal network (RPN)
  - an 'anchor pt' is placed at each column in feature map
  - each anchor pt generates k regions of fixed size & aspect ratio
  - in each region, predict object class & bounding box transform
- for each image:
  - run backbone CNN to get feature map
  - compute region proposals from RPN
- for each region:
  - crop & resize features
  - predict obj class & bbox transform
- ### training
  - RPN loss is weighted sum of:
    - classification loss: binary cross entropy loss
    - regression loss: SmoothL1 between true & predicted bbox parameters
  - training samples R region from N images
    - anchors sampled so up to 50% are objects
  - full network is RPN + Fast R-CNN (sharing a backbone)
    - various ways to train this, but ori method alternates between training RPN & Fast R-CNN
- learn region proposals using region proposal network (RPN)
- even faster than fast R-CNN (10x faster test)
- variation on Faster R-CNN with deep backbone can be quite accurate
  - speed-accuracy trade-off, deeper network is slower

## Single-stage object detectors

- ### YOLO v1
  - main idea: instead of going through multiple steps (region proposals, region classification), predict a heatmap for each class directly in a CNN
  - output is a set of N class prob. maps + M bounding box parameter maps
  - loss is sum-squared error between true & predicted maps, with some wieghtings:
    - bbox location parameters get higher weight in loss
    - grid cells that dont contain objects dont contribute to classification loss
    - bbox parameters penalised based on confidence, encourage M bboxes to specialise for diff. objects
  - #### advantages
    - fast
    - accurate, for real time object detector
  - #### disadvantages
    - limited spatial precision
    - less accurate than slower detectors
- ### SSD: single shot multibox detector
  - similar to YOLO, instead of generating region proposals, directly predict a set of class + bbox heatmaps
    - for each anchor pts: k bboxes (N class confidences, 4 bbox parameters)
  - major change: anchor pts in multiple convol layers, allowing for detection at diff. scales
  - faster than region-proposal methods like Faster R-CNN
  - less accurate than region-proposal methods
  - anchor pts in early layers helps with spatial prediction & detection of small objects
- skip region proposal step & predict obj classes/bbox directly

## Instance segmentation

- semantic segmentation classifies pixels, dont distinguish between instances
- common method:
  - run obj detector, extract bboxes & labels
  - do binary segmentation (foreground/background) within each bbox
- commonly used architecture: Mask R-CNN
- ### Mask R-CNN
  - extra step on Faster R-CNN
    - each patch runs thruogh a fully-convol network that predicts a binary segmentation mask
  - patch loss becomes L=Lcls + Lbox + Lmask
- can be modelled as obj detection followed by binary segmentation

## Object detection result

- typically obj detectors return many overlapping detections
- how to treat multiple detections?
- ### Non-max supression (NMS)
  - algo:
    - start with highest scoring bbox
    - drop bbox with lower score that overlap with this bbox greater than some IoU threshold (ex: 0.7)
    - repeat with next highest scoring bbox
  - often done seperately within each obj class
  - can drop some correct detections when objs are highly overlapping
  - generally is preferable to counting same objects many times
  - #### evaluation
    - run detection on entire test set
    - run NMS to remove overlapping detections
    - for each obj category, compute Average Precision (AP) = area under precision-recall (P-R) curve
    - mean average precision (mAP)
      - average AP over all object classes
    - COCO mAP
      - compute mAP for multiple IoU threshold and average

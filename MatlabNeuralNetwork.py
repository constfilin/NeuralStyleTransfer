import scipy.io
import numpy as np
import tensorflow as tf

class MatlabNeuralNetwork:
    """
    This class attempts to be compatible with many formats of VGG19 networks. It does not
    carry information about the layers in the network. This is captured in load_network
    """
    def __init__( self, path ):
        self.vgg    = scipy.io.loadmat(path)
        self.layers = self.vgg['layers']
        self.means  = None

    def get_means( self ):
        """
        Figure out "average image" from the mat file
        """
        if self.means is not None:
            return self.means
        # Ideally we need to modify scipy.io.loadmat code
        # Work with different formats of the network
        if all(i in self.vgg for i in ('layers', 'classes', 'normalization')):
            # Came from https://github.com/anishathalye/neural-style/blob/master/vgg.py#L26
            means = np.mean(self.vgg['normalization'][0][0][0], axis=(0, 1))
            means = np.reshape(means,(1,1,)+means.shape)
        elif all(i in self.vgg for i in ('layers','meta')):
            # Inspect dtypes (doing it the hard way)
            meta = self.vgg['meta']
            try: 
                normalization_ndx = list(meta.dtype.fields.keys()).index('normalization')
            except:
                raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
            normalization = meta[0][0][normalization_ndx]
            try:
                averageImage_ndx = list(normalization.dtype.fields.keys()).index('averageImage')
            except:
                raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
            means = normalization[0][0][averageImage_ndx]
        if means.shape!=(1,1,3):
            raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
        self.means = np.reshape(means,((1,) + means.shape))
        return self.means

    def get_layer( self, layer_ndx, expected_layer_name ):
        layer   = self.layers[0][layer_ndx][0][0]
        wb      = layer[2][0]
        result = {
            'name'    : layer[0][0],
            'weights' : wb[0],
            'bias'    : wb[1]
        }
        assert layer['name']==expected_layer_name
        return result

    def normalize_image( self, image ):
        # Normalize by substracting the mean to match the expected input of VGG19
        return image-self.get_means()

    def denormalize_image( self, image ):
        return image+self.get_means()

def get_nst_network(network,input_shape):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
         0 is conv1_1 (3, 3, 3, 64)
         1 is relu
         2 is conv1_2 (3, 3, 64, 64)
         3 is relu    
         4 is maxpool
         5 is conv2_1 (3, 3, 64, 128)
         6 is relu
         7 is conv2_2 (3, 3, 128, 128)
         8 is relu
         9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    def _conv2d( prev_layer, layer ):
        """
        Return the Conv2D layer using the weights, biases from the VGG model at 'layer'.
        """
        W = tf.constant(layer['weights'])
        b = tf.constant(np.reshape(layer['bias'],(layer['bias'].size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']    = tf.Variable(np.zeros(input_shape),dtype='float32',name='input')
    graph['conv1_1']  = tf.nn.relu(_conv2d(graph['input'],   network.get_layer(0,'conv1_1')))
    graph['conv1_2']  = tf.nn.relu(_conv2d(graph['conv1_1'], network.get_layer(2,'conv1_2')))
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = tf.nn.relu(_conv2d(graph['avgpool1'],network.get_layer(5,'conv2_1')))
    graph['conv2_2']  = tf.nn.relu(_conv2d(graph['conv2_1'], network.get_layer(7, 'conv2_2')))
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = tf.nn.relu(_conv2d(graph['avgpool2'],network.get_layer(10, 'conv3_1')))
    graph['conv3_2']  = tf.nn.relu(_conv2d(graph['conv3_1'], network.get_layer(12, 'conv3_2')))
    graph['conv3_3']  = tf.nn.relu(_conv2d(graph['conv3_2'], network.get_layer(14, 'conv3_3')))
    graph['conv3_4']  = tf.nn.relu(_conv2d(graph['conv3_3'], network.get_layer(16, 'conv3_4')))
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = tf.nn.relu(_conv2d(graph['avgpool3'],network.get_layer(19, 'conv4_1')))
    graph['conv4_2']  = tf.nn.relu(_conv2d(graph['conv4_1'], network.get_layer(21, 'conv4_2')))
    graph['conv4_3']  = tf.nn.relu(_conv2d(graph['conv4_2'], network.get_layer(23, 'conv4_3')))
    graph['conv4_4']  = tf.nn.relu(_conv2d(graph['conv4_3'], network.get_layer(25, 'conv4_4')))
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = tf.nn.relu(_conv2d(graph['avgpool4'],network.get_layer(28, 'conv5_1')))
    graph['conv5_2']  = tf.nn.relu(_conv2d(graph['conv5_1'], network.get_layer(30, 'conv5_2')))
    graph['conv5_3']  = tf.nn.relu(_conv2d(graph['conv5_2'], network.get_layer(32, 'conv5_3')))
    graph['conv5_4']  = tf.nn.relu(_conv2d(graph['conv5_3'], network.get_layer(34, 'conv5_4')))
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

def shape_image( image ):
    return np.reshape(image,((1,)+image.shape))

def deshape_image( image ):
    return np.clip(image[0],0,255).astype('uint8')

def generate_noise_image(image,noise_ratio):
    """
    Generates a noisy image by adding random noise to the image
    """    
    # Generate a random noise_image
    noise_image = np.random.uniform(-20,20,(1,image.shape[1],image.shape[2],image.shape[3])).astype('float32')
    # Set the input_image to be a weighted average of the image and a noise_image
    return noise_image * noise_ratio + image * (1 - noise_ratio)


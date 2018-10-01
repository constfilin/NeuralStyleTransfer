#!/home/cf/Work/DeepLearning/bin/python

import os
import sys
import re
import time
import argparse 

import skimage.io
import tensorflow as tf
import MatlabNeuralNetwork as mnn

def compute_content_cost(sess,input_image,layer):
    """
    sess            - TF session
    input_image     - the content image
    layer           - the content layer (Tensor)
    """
    # Evaluate the session on input image (should be the content image). 
    # By doing this we - basically - tell the network to run on the initial value of content image
    # Without running the session on input image we get error 'Attempting to use uninitialized value input'
    tmp = sess.run(graph['input'].assign(input_image))
    # Now we run 
    evaluated_layer  = sess.run(layer)
    # Reshape the images to have them of shape (n_C, n_H*n_W)
    m, n_H, n_W, n_C = layer.get_shape().as_list()
    evaluated_layer_unrolled = tf.reshape(tf.transpose(evaluated_layer),[-1,n_C])
    layer_unrolled           = tf.reshape(tf.transpose(layer),[-1,n_C])
    return  2*tf.nn.l2_loss(layer_unrolled-evaluated_layer_unrolled)/(4*n_H*n_W*n_C)

def compute_style_cost(sess,input_image,layer_coeffs):
    def compute_layer_style_cost(sess,layer):
        evaluated_layer = sess.run(layer)
        # Reshape the images to have them of shape (n_C, n_H*n_W)
        m, n_H, n_W, n_C = layer.get_shape().as_list()
        evaluated_layer_unrolled = tf.reshape(tf.transpose(evaluated_layer),[n_C,-1])
        layer_unrolled           = tf.reshape(tf.transpose(layer),[n_C,-1])
        # Computing gram_matrices for both images S and G (≈2 lines), see http://prntscr.com/l0orqq
        evaluated_gram = tf.matmul(evaluated_layer_unrolled,tf.transpose(evaluated_layer_unrolled))
        gram           = tf.matmul(layer_unrolled,tf.transpose(layer_unrolled))
        return 2*tf.nn.l2_loss(gram-evaluated_gram)/(4*(n_C*n_H*n_W)**2)
    # Evaluate the session on input image (should be the style image)
    tmp = sess.run(graph['input'].assign(input_image))
    # initialize the overall style cost
    style_cost = 0
    for layer,coeff in layer_coeffs:
         style_cost += coeff*compute_layer_style_cost(sess,layer)
    return style_cost

def print_progress(i):
    folder_name = "%s_@%02dx%02d_@%04d_@%dx%d" % (
        output_image_matches.group(1),
        int(args.content_weight),int(50-args.content_weight),
        int(args.noise_ratio*100),
        style_image.shape[2],style_image.shape[1])
    try:
        os.mkdir(folder_name)
    except:
        # print("Cannot create folder %s (%s)" % (folder_name,sys.exc_info()[0]))
        pass
    now        = time.time()
    speed      = (now-start)/(i if i>0 else 1)
    Jt, Jc, Js = sess.run([total_cost,content_cost,style_cost])
    print("Iteration %4d costs: total=%.2f,content=%.2f,style=%.2f, speed=%.2f iterations/sec, ETA=%s" % (
        i,
        Jt,
        Jc,
        Js,
        speed,
        time.ctime(start+speed*args.iterations)))
    generated_image = sess.run(graph["input"])
    skimage.io.imsave(
        "%s/%04d.%s" % (folder_name,i,output_image_matches.group(2)),
        mnn.deshape_image(network.denormalize_image(generated_image)))

########################################################################################################    
# TOP LEVEL
########################################################################################################
parser = argparse.ArgumentParser(description="Do neural style transfer from style image and content image into output image")
parser.add_argument('--style'         ,help='Style image',required=True)
parser.add_argument('--content'       ,help='Content image',required=True)
parser.add_argument('--output'        ,help='Output image',required=True)
parser.add_argument('--network'       ,help='Pre-trained VGG network MATLAB file',default="pretrained-model/imagenet-vgg-verydeep-19.mat")
parser.add_argument('--content_layer' ,help='Content layer name',default='conv4_2')
parser.add_argument('--noise_ratio'   ,help='Noise applied to the content image',default=0.0,type=float)
parser.add_argument('--iterations'    ,help='Number of iterations',default=160,type=int)
parser.add_argument('--checkpoints'   ,help='How many iteractions have to pass between saving checkpoint images',default=20,type=int)
parser.add_argument('--content_weight',help='Weight of the content image in the output, from 0 to 50',default=1.0,type=float)
parser.add_argument('--learning_rate' ,help='Learning rate of Adam optimizer',default=2.0,type=float)
args = parser.parse_args()

output_image_matches = re.match('(^[^\.]+)\.(jpg|png)$',args.output)
if not output_image_matches:
    print('Output file name should satisfy RegExp (^[^\.]+)\.(jpg|png)$')
    sys.exit(2)

network       = mnn.MatlabNeuralNetwork(args.network)
style_image   = network.normalize_image(mnn.shape_image(skimage.io.imread(args.style)))
content_image = network.normalize_image(mnn.shape_image(skimage.io.imread(args.content)))
if style_image.shape!=content_image.shape:
    print("The shape of %s and %s are different (%s!=%s)" % (args.style,args.content,style_image.shape,content_image.shape))
    sys.exit(1)

tf.reset_default_graph()
sess         = tf.Session()
graph        = mnn.get_nst_network(network,style_image.shape)
content_cost = args.content_weight*compute_content_cost(sess,content_image,graph[args.content_layer])
style_cost   = (50-args.content_weight)*compute_style_cost(sess,
                                                           style_image,
                                                           [(graph['conv1_1'], 0.2),
                                                            (graph['conv2_1'], 0.2),
                                                            (graph['conv3_1'], 0.2),
                                                            (graph['conv4_1'], 0.2),
                                                            (graph['conv5_1'], 0.2)])
# Now define what we want to optimize
total_cost = content_cost+style_cost
train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(total_cost)

# Initialize global variables (you need to run the session on the initializer)
sess.run(tf.global_variables_initializer())
# Run the noisy input image (initial generated image) through the graph. Use assign().
sess.run(graph["input"].assign(mnn.generate_noise_image(content_image,args.noise_ratio)))
start = time.time()

for i in range(args.iterations):
    # Run the session on the train_step to minimize the total cost
    sess.run(train_step)
    # Compute the generated image by running the session on the current graph['input']
    if (i%args.checkpoints)==0:
        print_progress(i)

print_progress(args.iterations)


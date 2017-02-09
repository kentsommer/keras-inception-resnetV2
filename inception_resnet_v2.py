# Sys
import warnings
# Keras Core
from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# Backend
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

#########################################################################################
# Implements the Inception ResNet v2 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################

# TH_WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5'
# TF_WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

def do_scale(x, scale):
    y = scale * x 
    return y 


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), 
              bias=False, activ_fn='relu', normalize=True):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    if not normalize:
        bias = True
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias)(x)
    if normalize:
        x = BatchNormalization(axis=channel_axis)(x)
    if activ_fn:
        x = Activation(activ_fn)(x)
    return x


def block35(input, scale=1.0, activation_fn='relu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 32, 3, 3, activ_fn=activation_fn)

    tower_conv2_0 = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)
    tower_conv2_1 = conv2d_bn(tower_conv2_0, 48, 3, 3, activ_fn=activation_fn)
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 64, 3, 3, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 320, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net


def block17(input, scale=1.0, activation_fn='relu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 128, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 160, 1, 7, activ_fn=activation_fn)
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 192, 7, 1, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 1088, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net


def block8(input, scale=1.0, activation_fn='relu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 224, 1, 3, activ_fn=activation_fn)
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 256, 3, 1, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 2080, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net


def inception_resnet_v2(num_classes, dropout_keep_prob, weights):
    '''
    Creates the inception_resnet_v2 network

    Args:
        num_classes: number of classes
        dropout_keep_prob: float, the fraction to keep before final layer.
    Returns: 
        logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 299, 299))
        channel_axis = 1
    else:
        inputs = Input((299, 299, 3))
        channel_axis = -1

    #########################
    # Build Base of network #
    #########################

    # 149 x 149 x 32
    net = conv2d_bn(inputs, 32, 3, 3, subsample=(2,2), border_mode='valid')
    # 147 x 147 x 32
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    # 147 x 147 x 64
    net = conv2d_bn(net, 64, 3, 3)
    # 73 x 73 x 64
    net = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)
    # 73 x 73 x 80
    net = conv2d_bn(net, 80, 1, 1, border_mode='valid')
    # 71 x 71 x 192
    net = conv2d_bn(net, 192, 3, 3, border_mode='valid')
    # 35 x 35 x 192
    net = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)


    ## Tower One
    tower_conv = conv2d_bn(net, 96, 1, 1)

    tower_conv1_0 = conv2d_bn(net, 48, 1, 1)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 64, 5, 5)

    tower_conv2_0 = conv2d_bn(net, 64, 1, 1)
    tower_conv2_1 = conv2d_bn(tower_conv2_0, 96, 3, 3)
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 96, 3, 3)

    tower_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(net)
    tower_pool_1 = conv2d_bn(tower_pool, 64, 1, 1)

    net = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], mode='concat', concat_axis=channel_axis)

    # 10 x block35
    for idx in xrange(10):
        net = block35(net, scale=0.17)


    ## Tower Two
    tower_conv = conv2d_bn(net, 384, 3, 3, subsample=(2,2), border_mode='valid')

    tower_conv1_0 = conv2d_bn(net, 256, 1, 1)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 256, 3, 3)
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 384, 3, 3, subsample=(2,2), border_mode='valid')

    tower_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(net)

    net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', concat_axis=channel_axis)

    # 20 x block17
    for idx in xrange(20):
        net = block17(net, scale=0.10)


    ## Tower Three
    tower_conv = conv2d_bn(net, 256, 1, 1)
    tower_conv_1 = conv2d_bn(tower_conv, 384, 3, 3, subsample=(2,2), border_mode='valid')

    tower_conv1 = conv2d_bn(net, 256, 1, 1)
    tower_conv1_1 = conv2d_bn(tower_conv1, 288, 3, 3, subsample=(2,2), border_mode='valid')

    tower_conv2 = conv2d_bn(net, 256, 1, 1)
    tower_conv2_1 = conv2d_bn(tower_conv2, 288, 3, 3)
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

    tower_pool = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    net = merge([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], mode='concat', concat_axis=channel_axis)

    # 9 x block8
    for idx in xrange(9):
        net = block8(net, scale=0.20)
    net = block8(net, activation_fn=False)


    # Logits
    net = conv2d_bn(net, 1536, 1, 1)
    net = AveragePooling2D((8,8), border_mode='valid')(net)

    net = Flatten()(net)
    net = Dropout(dropout_keep_prob)(net)

    predictions = Dense(output_dim=num_classes, activation='softmax')(net)

    model = Model(inputs, predictions, name='inception_resnet_v2')

    if weights:
    	model.load_weights(weights, by_name=True)
    	print("Loaded Model Weights!")

    return model

def create_model(num_classes=1001, dropout_keep_prob=0.8, weights=None):
	return inception_resnet_v2(num_classes, dropout_keep_prob, weights)




    
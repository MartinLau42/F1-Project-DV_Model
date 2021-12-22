from build_model_for_single_tephi_input import *
from preprocess_for_multi_tephi_input import *

# For 2-tephi model, create_model_(..., img_size=(2, 350, 250, 3), data_shape=(2, 90, 5), ...)
# For 4-tephi model, create_model_(..., img_size=(4, 350, 250, 3), data_shape=(4, 90, 5), ...)

#ReLU = tf.keras.layers.LeakyReLU(alpha=0.01)
ReLU = 'relu'

def build_conv2d(x_rescale, pretrain):
    x = pretrain(x_rescale, training=False)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x) # name='dropout'
    x = layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x) # name='dense'
    return x


def build_conv1d(x_input):
    x = layers.Conv1D(8, 3, activation=ReLU)(x_input) # name='conv1d'
    x = layers.MaxPooling1D(2)(x) # name='max_pooling1d'
    x = layers.Conv1D(16, 3, activation=ReLU)(x) # name='conv1d_1'
    x = layers.MaxPooling1D(2)(x) # name='max_pooling1d_1'
    x = layers.Conv1D(32, 3, activation=ReLU)(x) # name='conv1d_2'
    x = layers.Conv1D(32, 3, activation=ReLU)(x) # name='conv1d_3'
    x = layers.MaxPooling1D(2)(x) # name='max_pooling1d_2'
    x = layers.Flatten()(x)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='tanh')(x) # name='dense_1'
    return x


def create_model_(pretrain, pretrain_name, n_class, img_size=(2, 350, 250, 3), data_shape=(2, 90, 5), initial_bias=None):
    if pretrain_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif pretrain_name == 'vgg19':
        from tensorflow.keras.applications.vgg19 import preprocess_input
    elif pretrain_name == 'xception':
        from tensorflow.keras.applications.xception import preprocess_input
    elif pretrain_name == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif pretrain_name == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif pretrain_name == 'resnet_v2':
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif pretrain_name == 'inception_resnet_v2':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    elif pretrain_name == 'mobilenet':
        from tensorflow.keras.applications.mobilenet import preprocess_input
    else:
        raise Exception('This pretrain_name is not imported')
    
    pretrain.trainable = False

    x1_input = layers.Input(shape=img_size, dtype=tf.uint8)
    
    x1_input_a = x1_input[:, 0]
    x1_rescale_a = preprocess_input(x1_input_a)
    x1a = build_conv2d(x1_rescale_a, pretrain)
    x1a = layers.Reshape((1, -1))(x1a)

    x1_input_b = x1_input[:, 1]
    x1_rescale_b = preprocess_input(x1_input_b)
    x1b = build_conv2d(x1_rescale_b, pretrain)
    x1b = layers.Reshape((1, -1))(x1b)
    # Comment out if want to train 4-tephi model
    '''
    x1_input_c = x1_input[:, 2]
    x1_rescale_c = preprocess_input(x1_input_c)
    x1c = build_conv2d(x1_rescale_c, pretrain)
    x1c = layers.Reshape((1, -1))(x1c)

    x1_input_d = x1_input[:, 3]
    x1_rescale_d = preprocess_input(x1_input_d)
    x1d = build_conv2d(x1_rescale_d, pretrain)
    x1d = layers.Reshape((1, -1))(x1d)
    '''
    concatenated_1 = layers.Concatenate(axis=1)([x1a, x1b]) #, x1c, x1d])
    LSTM_1 = layers.LSTM(128,
                         #dropout=0.2, 
                         #recurrent_dropout=0.2,
                         input_shape=(2, None))(concatenated_1) # 4

    x2_input = layers.Input(shape=data_shape, dtype=tf.float64) # TT, Td, Wdir, Wspd & depression

    x2_input_a = x2_input[:, 0]
    x2a = build_conv1d(x2_input_a)
    x2a = layers.Reshape((1, -1))(x2a)

    x2_input_b = x2_input[:, 1]
    x2b = build_conv1d(x2_input_b)
    x2b = layers.Reshape((1, -1))(x2b)
    # Comment out if want to train 4-tephi model
    '''
    x2_input_c = x2_input[:, 2]
    x2c = build_conv1d(x2_input_c)
    x2c = layers.Reshape((1, -1))(x2c)

    x2_input_d = x2_input[:, 3]
    x2d = build_conv1d(x2_input_d)
    x2d = layers.Reshape((1, -1))(x2d)
    '''
    concatenated_2 = layers.Concatenate(axis=1)([x2a, x2b]) #, x2c, x2d])
    LSTM_2 = layers.LSTM(32,
                         #dropout=0.2,
                         #recurrent_dropout=0.2,
                         input_shape=(2, None))(concatenated_2) # 4

    concatenated = layers.Concatenate()([LSTM_1, LSTM_2])
    concatenated = layers.Dense(10, activation=ReLU)(concatenated)
    if n_class == 3:
        output = layers.Dense(3, activation='softmax')(concatenated)
    elif n_class == 2:
        output = layers.Dense(1, activation='sigmoid',
                              bias_initializer=initializers.Constant(initial_bias))(concatenated)
    model = models.Model([x1_input, x2_input], output)
    return model

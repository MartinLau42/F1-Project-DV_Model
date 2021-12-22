from preprocess_for_single_tephi_input import *
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import Xception, VGG16, VGG19, ResNet50, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet

#ReLU = tf.keras.layers.LeakyReLU(alpha=0.01)
ReLU = 'relu'

def create_model(pretrain, pretrain_name, n_class, img_size=(350, 250, 3), data_shape=(90, 5), initial_bias=None):
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
    x1_input = layers.Input(shape=img_size, dtype=tf.float32)
    x1_rescale = preprocess_input(x1_input)
    x = pretrain(x1_rescale, training=False)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x)

    x2_input = layers.Input(shape=data_shape) # TT, Td, Wdir, Wspd & depression
    x2 = layers.Conv1D(8, 3, activation='relu')(x2_input)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Conv1D(16, 3, activation='relu')(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Conv1D(32, 3, activation='relu')(x2)
    x2 = layers.Conv1D(32, 3, activation='relu')(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(32, activation='tanh')(x2) # kernel_regularizer=regularizers.l2(0.01)

    concatenated = layers.Concatenate()([x, x2])
    concatenated = layers.Dropout(0.5)(concatenated)
    if n_class == 3:
        output = layers.Dense(3, activation='softmax')(concatenated)
    elif n_class == 2:
        output = layers.Dense(1, activation='sigmoid',
                      bias_initializer=initializers.Constant(initial_bias))(concatenated)
    model = models.Model([x1_input, x2_input], output)
    return model


def create_model_2(n_class, img_size=(350, 250, 3), data_shape=(90, 5), initial_bias=None):
    from tensorflow.keras.applications.vgg19 import preprocess_input
    pretrain = VGG19(include_top=False,
                     weights="imagenet",
                     input_shape=img_size)
    
    x1_input = layers.Input(shape=img_size, dtype=tf.float32)
    x1_rescale = preprocess_input(x1_input)
    
    # Self-build VGG19 in order to be able to plot Class Activation Map (CAM) for its intermediate layers
    # Block 1
    x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
          x1_rescale)
    x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    # Custom layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=ReLU, kernel_regularizer=regularizers.l2(0.001))(x) # 'tanh'

    x2_input = layers.Input(shape=data_shape) # TT, Td, Wdir, Wspd & depression
    x2 = layers.Conv1D(8, 3, activation=ReLU)(x2_input)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Conv1D(16, 3, activation=ReLU)(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Conv1D(32, 3, activation=ReLU)(x2)
    x2 = layers.Conv1D(32, 3, activation=ReLU)(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(32, activation=ReLU)(x2) # kernel_regularizer=regularizers.l2(0.01), activation='tanh'

    concatenated = layers.Concatenate()([x, x2])
    concatenated = layers.Dropout(0.5)(concatenated)
    if n_class == 3:
        output = layers.Dense(3, activation='softmax')(concatenated)
    elif n_class == 2:
        output = layers.Dense(1, activation='sigmoid',
                      bias_initializer=initializers.Constant(initial_bias))(concatenated)
    model = models.Model([x1_input, x2_input], output)
    
    # Transfer pretrained VGG19 weights to the self-build layers
    pretrain_layer_names = [l.name for l in pretrain.layers]
    for l in model.layers:
        if l.name in pretrain_layer_names:
            orig_l = pretrain.get_layer(l.name)
            l.set_weights(orig_l.get_weights())
            
    # Freeze all pretrain layers
    print('Number of ALL trainable weights:', len(model.trainable_weights))
    model.trainable = True
    for layer in model.layers:
        if layer.name in pretrain_layer_names:
            layer.trainable = False
            
    print('Number of trainable weights after freezing all pretrain layers:', len(model.trainable_weights))        
    return model, pretrain_layer_names


def compile_model(model, n_class, lr=1e-3):
    if n_class >= 3:
        METRICS = [metrics.SparseCategoricalAccuracy(name='accuracy')]
        loss = 'sparse_categorical_crossentropy'
    elif n_class == 2:
        METRICS = [
                    metrics.Precision(name='precision'),
                    metrics.Recall(name='recall'),
                    metrics.AUC(curve='ROC', name='auc'), 
                    metrics.AUC(curve='PR', name='prc'),
                    metrics.BinaryAccuracy(name='accuracy')
                    ]
        loss = 'binary_crossentropy'
    #optimizer = optimizers.RMSprop(learning_rate=lr, centered=True, rho=0.9)
    optimizer = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)

    
def get_callback_list(monitor='val_loss', min_delta=0.01, patience=40, factor=None, checkpoint_dir=None, HHMM=None):
    EarlyStopping = callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, restore_best_weights=True)
    
    if factor is not None:
        ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, min_delta=0.0001, min_lr=0, patience=10)
    else:
        ReduceLROnPlateau = None
        
    if (checkpoint_dir is not None) & (HHMM is not None):
        ModelCheckpoint = callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'model.{epoch:03d}-{val_loss:.2f}_%s' % HHMM), 
                                                    monitor=monitor, period=1)
        CSVLogger = callbacks.CSVLogger(os.path.join(checkpoint_dir, 'epoch_%s_tuned.csv' % HHMM), separator=',', append=True)
    else:
        ModelCheckpoint = CSVLogger = None
    
    callback_list = [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger]
    callback_list = [c for c in callback_list if c is not None]
    return callback_list


def get_initial_bias(y_train):
    if len(np.unique(y_train)) == 2:
        pos = np.sum(y_train)
        neg = np.sum(1-y_train)
        initial_bias = np.log([pos/neg])
        return initial_bias
    else:
        raise Exception('initial_bias can only simply be calculated under binary classification')

    
def get_class_weight(y_train):
    y_train = np.asarray(y_train)
    unique_class = np.unique(y_train)
    n_classes = len(unique_class)
    total = len(y_train)
    class_weight = {}
    for i in unique_class:
        class_weight[i] = (1 / np.sum(y_train==i)) * (total / n_classes)
    return class_weight


def fine_tune(pretrain, trainable_layer='block5_conv1'):
    # Unfreeze layers including and after 'block5_conv1'
    set_trainable = False

    pretrain.trainable = True
    for layer in pretrain.layers:
        if layer.name == trainable_layer: 
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

            
def fine_tune_2(model, pretrain_layer_names, num_set_trainable=5):
    # Unfreeze the last 5 layers by default
    model_pretrain_layer = [l for l in model.layers if l.name in pretrain_layer_names]
    model_pretrain_layer_name = [l.name for l in model_pretrain_layer]
    for layer in model_pretrain_layer:
        layer.trainable = True

    layers_set_trainable = model_pretrain_layer_name[-num_set_trainable:]
    for layer in model_pretrain_layer:
        if layer.name in layers_set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
            
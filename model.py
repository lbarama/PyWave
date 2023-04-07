import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, Activation
import matplotlib.pyplot as plt



#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# test to see if tensorflow will use the GPU
gpu_use = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


def build_model_3channel(dropout_value=0, wave_length=1200):
    # test to see if tensorflow will use the GPU
    gpu_use = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print('GPU use: {}'.format(gpu_use))

    model = Sequential()

    # Layer 1, 2000 -> 1000
    model.add(Conv1D(8, kernel_size=5,strides=1, padding='same',
        input_shape=(wave_length,3), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 2, 1000 -> 500
    model.add(Conv1D(16, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 3, 500 -> 250
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))
    
    # Layer 4, 250 -> 127
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 5, 127 -> 64
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 6, 64 -> 32
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 7, 32 -> 16
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 8, 16 -> 8
    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 9, 8 -> 4
    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model_1channel(dropout_value=0, wave_length=1200):
    # test to see if tensorflow will use the GPU
    gpu_use = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print('GPU use: {}'.format(gpu_use))

    model = Sequential()

    # Layer 1, 2000 -> 1000
    model.add(Conv1D(8, kernel_size=5,strides=1, padding='same',
        input_shape=(wave_length,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 2, 1000 -> 500
    model.add(Conv1D(16, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 3, 500 -> 250
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))
    
    # Layer 4, 250 -> 127
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 5, 127 -> 64
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 6, 64 -> 32
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 7, 32 -> 16
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 8, 16 -> 8
    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 9, 8 -> 4
    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def plot_results(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    # Plot training & validation accuracy values on prescribed y-scale
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(.8,1)
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


def build_model_1channel_3class(dropout_value=0, wave_length=400):
    # test to see if tensorflow will use the GPU
    gpu_use = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print('GPU use: {}'.format(gpu_use))

    model = Sequential()

    # Layer 1, 2000 -> 1000
    model.add(Conv1D(8, kernel_size=5,strides=1, padding='same',
        input_shape=(wave_length,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 2, 1000 -> 500
 #   model.add(Conv1D(16, kernel_size=5, strides=1, padding='same', use_bias=False))
 #   model.add(BatchNormalization(axis=1))
 #   model.add(Activation("relu"))
 #   model.add(Dropout(dropout_value))
 #   model.add(MaxPooling1D(2))

    # Layer 3, 500 -> 250
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))
    
    # Layer 4, 250 -> 127
#   model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Layer 5, 127 -> 64
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 6, 64 -> 32
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 7, 32 -> 16
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 8, 16 -> 8
#    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Layer 9, 8 -> 4
#    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model_1channel_4class(dropout_value=0, wave_length=400):
    # test to see if tensorflow will use the GPU
    gpu_use = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print('GPU use: {}'.format(gpu_use))

    model = Sequential()

    # Layer 1, 2000 -> 1000
    model.add(Conv1D(8, kernel_size=6,strides=1, padding='same',
        input_shape=(wave_length,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 2, 1000 -> 500
 #   model.add(Conv1D(16, kernel_size=5, strides=1, padding='same', use_bias=False))
 #   model.add(BatchNormalization(axis=1))
 #   model.add(Activation("relu"))
 #   model.add(Dropout(dropout_value))
 #   model.add(MaxPooling1D(2))

    # Layer 3, 500 -> 250
    model.add(Conv1D(16, kernel_size=4, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))
    
    # Layer 4, 250 -> 127
#   model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Layer 5, 127 -> 64
    model.add(Conv1D(32, kernel_size=4, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 6, 64 -> 32
    model.add(Conv1D(16, kernel_size=4, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 7, 32 -> 16
    model.add(Conv1D(16, kernel_size=4, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_value))
    model.add(MaxPooling1D(2))

    # Layer 8, 16 -> 8
#    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Layer 9, 8 -> 4
#    model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', use_bias=False))
#    model.add(BatchNormalization(axis=1))
#    model.add(Activation("relu"))
#    model.add(Dropout(dropout_value))
#    model.add(MaxPooling1D(2))

    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

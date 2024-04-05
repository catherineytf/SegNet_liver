import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

image_directory = 'MRI/Anatomical_mag_echo5/'
mask_directory = 'MRI/whole_liver_segmentation/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []
image_filenames =[]
image_slices_filenames =[]
mask_filenames =[]
mask_slices_filenames =[]


images = os.listdir(image_directory)
images.sort()
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image_dataset.append(np.array(image))


        image_filenames.append(os.path.splitext(image_name)[0]) 

masks = os.listdir(mask_directory)
masks.sort()
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        mask_dataset.append(np.array(image))

        mask_filenames.append(os.path.splitext(image_name)[0]) 


for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        
        image_slice_id = f'{image_filenames[i]}-slice{j}'
        image_slices_filenames.append(image_slice_id)


for i in range(len(mask_dataset)):
    for j in range(mask_dataset[i].shape[2]):
        if mask_filenames[i] == 'f_3325' and j==31:
            continue
        else:
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])
            mask_slice_id = f'{mask_filenames[i]}-slice{j}'
            mask_slices_filenames.append(mask_slice_id)


print(f"Total sliced images: {len(sliced_image_dataset)}")
print(f"Total sliced masks: {len(sliced_mask_dataset)}")

batch_size = 16 # The batch size to use when training the model
image_size = (224, 224)  # The size of the images

import tensorflow as tf
import tifffile as tiff
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Concatenate

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Building the SegNet Model



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def load_pretrained_vgg16(input_shape, weights_path):
    """
    Loads VGG16 model as the encoder with pre-trained ImageNet weights.
    
    Args:
    - input_shape: Shape of the input images (height, width, channels).
    - weights_path: Path to the pre-trained weights file.
    
    Returns:
    - Pre-trained VGG16 model without the top layer.
    """
    base_model = VGG16(include_top=False, weights=None, input_tensor=Input(shape=input_shape))
    base_model.load_weights(weights_path)
    return base_model

def decoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, dropout_rate=0.1):
    """
    Decoder block for SegNet, performing upsampling and convolution.
    
    Args:
    - input_tensor: Input tensor for the decoder block.
    - n_filters: Number of filters for the convolutional layers.
    - kernel_size: Size of the kernel for the convolutional layers.
    - batchnorm: Whether to include BatchNormalization layers.
    - dropout_rate: Dropout rate.
    
    Returns:
    - Output tensor of the decoder block.
    """
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_segnet_model(input_shape=(224, 224, 3), n_classes=1, dropout_rate=0.1, weights_path='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'):
    """
    Builds the SegNet model using a pre-trained VGG16 encoder and a custom decoder.
    
    Args:
    - input_shape: Shape of the input images.
    - n_classes: Number of output classes.
    - dropout_rate: Dropout rate for the decoder.
    - weights_path: Path to the VGG16 pre-trained weights.
    
    Returns:
    - SegNet model built with TensorFlow/Keras.
    """
    # Load the pre-trained VGG16 model as the encoder
    encoder = load_pretrained_vgg16(input_shape, weights_path)
    
    # Freeze the encoder layers
    for layer in encoder.layers:
        layer.trainable = False
    
    # Start building the decoder from the last layer of the encoder
    encoded = encoder.output
    
    # Add decoder blocks
    decoded = decoder_block(encoded, 512, dropout_rate=dropout_rate)
    decoded = decoder_block(decoded, 256, dropout_rate=dropout_rate)
    decoded = decoder_block(decoded, 128, dropout_rate=dropout_rate)
    decoded = decoder_block(decoded, 64, dropout_rate=dropout_rate)
    decoded = UpSampling2D(size=(2, 2), interpolation='bilinear')(decoded)  # Adjust size as necessary

    
    # Output layer
    output = Conv2D(n_classes, (1, 1), activation='sigmoid')(decoded)
    
    # Create model
    model = Model(inputs=encoder.input, outputs=output)
    
    return model

from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


import tensorflow as tf

def true_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_pos = tf.cast(y_true > threshold, tf.float32)
    
    # Use `tf.logical_and` directly from TensorFlow
    true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
    actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
    
    tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
    return tpr

def false_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_neg = tf.cast(y_true <= threshold, tf.float32)
    
    # Use `tf.logical_and` directly from TensorFlow
    false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
    actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
    
    fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
    return fpr


from sklearn.metrics import confusion_matrix

def calculate_tpr_fpr(y_true, y_pred):
    # Assuming y_pred is sigmoid output, threshold to get binary mask
    y_pred = y_pred > 0.5
    # Flatten the arrays to compute confusion matrix
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    cm = confusion_matrix(y_true_f, y_pred_f).ravel()
    
    # Depending on the shape of the confusion matrix, unpack accordingly
    if cm.shape[0] == 4:  # If we have a full 2x2 matrix
        tn, fp, fn, tp = cm
    elif cm.shape[0] == 1:  # If we only have one value, it means only one class was predicted
        # Check which class is present
        if np.unique(y_true_f).item() == 1:  # Only positives are present
            tp = cm[0]
            tn = fp = fn = 0
        else:  # Only negatives are present
            tn = cm[0]
            tp = fp = fn = 0
    else:  # This is for the case where the confusion matrix might be 2 elements long (only 2 out of tp, fp, tn, fn are present)
        raise ValueError("Unexpected confusion matrix shape.")
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Handling division by zero
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Handling division by zero
    
    return tpr, fpr


from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 30
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

lr_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)


from sklearn.model_selection import KFold

#change number of folds as needed 
n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=21)
histories = []
TPRs =[]
FPRs =[]
# Assuming each element in your lists is a NumPy array of the same shape
sliced_image_dataset = np.array(sliced_image_dataset)
sliced_mask_dataset = np.array(sliced_mask_dataset)



# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]
    all_test_indices = []
    all_test_indices.append(test_index)
    all_test_indices = np.array(all_test_indices)

    f = open("C:/Users/Mittal/Desktop/2D-segnet-liver-copy/test_index.txt", "a")
    np.set_printoptions(threshold=2000)
    f.write(f'fold{i}')
    for indices in all_test_indices:
        print(f'{indices}\n',file=f)
    
    f.close()


    X_train_expanded=np.expand_dims(X_train, axis=-1)
    X_test_expanded = np.expand_dims(X_test, axis=-1)

    X_train_rgb = np.repeat(X_train_expanded, 3, axis=-1)
    X_test_rgb = np.repeat(X_test_expanded, 3, axis=-1)

    # Now, X_train_rgb and X_test_rgb have shape (num_samples, 224, 224, 3) and can be used for training

   
    checkpoint = ModelCheckpoint(f'best_model{i}.keras', monitor='val_loss', save_best_only=True)
    # Model setup and training
    model = build_segnet_model(input_shape=(224, 224, 3), n_classes=1, dropout_rate=0.1, weights_path='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='bce', 
                  metrics=[dice_coef, true_positive_rate, false_positive_rate])



    history = model.fit(X_train_rgb, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=300,
                        validation_data=(X_test_rgb, y_test),
                        shuffle=False,
                        callbacks=[checkpoint])

 
    histories.append(history)


    #plot for loss 
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'])
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val.'], loc='upper right')
    
    #plot for dice
    plt.subplot(1,2,2)
    plt.plot(history.history['dice_coef'], color='r')
    plt.plot(history.history['val_dice_coef'])
    plt.ylabel('dice_coef')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'C:/Users/Mittal/Desktop/2D-segnet-liver-copy/process{i}.png')
    plt.close()


    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])
    max_tpr = max(history.history['true_positive_rate'])
    min_fpr = min(history.history['false_positive_rate'])
    max_val_tpr = max(history.history['true_positive_rate'])
    min_val_fpr = min(history.history['false_positive_rate'])


    f = open("C:/Users/Mittal/Desktop/2D-segnet-liver-copy/output.txt", "a")
    print(f'max_dice_coef:{max_dice_coef}', file=f)
    print(f'max_val_dice_coef:{max_val_dice_coef}', file=f)
    print(f'max_tpr:{max_tpr}', file=f)
    print(f'min_fpr:{min_fpr}', file=f)
    print(f'max_tpr:{max_val_tpr}', file=f)
    print(f'min_fpr:{min_val_fpr}', file=f)
    f.close()

    model.load_weights(f'C:/Users/Mittal/Desktop/2D-segnet-liver-copy/best_model{i}.keras')
    
    
    #5 test images 
    for z in range(10):
        

        test_img_number = random.randint(0, len(X_test))
        test_img = X_test[test_img_number]
        ground_truth = y_test[test_img_number]
        #test_img_norm = test_img[:,:,0][:,:,None]




        test_expand = np.expand_dims(test_img, axis=-1)
        test_rgb= np.repeat(X_train_expanded, 3, axis=-1)

        test_img_input = test_rgb
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

        original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
        colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
        alpha = 0.5 
        colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

      
        
        
        tpr, fpr = calculate_tpr_fpr(ground_truth, prediction)
        print(f'TPR={tpr}')
        print(f'FPR={fpr}')

        TPRs.append(tpr)
        FPRs.append(fpr)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(141)
        plt.title('Testing Image')
        plt.imshow(test_img[:,:], cmap='gray')
        plt.subplot(142)
        plt.title('Testing Mask')
        plt.imshow(ground_truth[:,:], cmap='gray')
        plt.subplot(143)
        plt.title('Prediction on test image')
        plt.imshow(prediction, cmap='gray')
        plt.subplot(144)
        plt.title("Overlayed Images")
        plt.imshow(original_image_normalized, cmap='gray')
        plt.imshow(colored_mask, cmap='jet')
        plt.savefig(f'C:/Users/Mittal/Desktop/2D-segnet-liver-copy/predict/fold{i}_{z}.png')
        plt.close()
        

    


    

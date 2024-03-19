pip install nibabel
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

#matching images with their masks
import os

base_dir = 'MRI/'
images_dir = os.path.join(base_dir, 'Anatomical_mag_echo5')
masks_dir = os.path.join(base_dir, 'whole_liver_segmentation')

# List comprehension to get full paths of files
image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii')]
mask_paths = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.nii')]

# Sort to ensure matching order
image_paths.sort()
mask_paths.sort()
# Check the number of images and labels gathered
print(f"Total images: {len(image_paths)}")
print(f"Total labels: {len(mask_paths)}")

import tensorflow as tf
import random

class NiiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, image_size):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size, *self.image_size, 1))
        y = np.zeros((self.batch_size, *self.image_size, 1))

        for i, (image_filename, mask_filename) in enumerate(zip(batch_x, batch_y)):
            image = nib.load(image_filename)
            mask = nib.load(mask_filename)
            # get the data from the image object
            image_data = image.get_fdata()
            mask_data = mask.get_fdata()
            # get random slice from the volumes
            slice_index = random.randint(0, image_data.shape[2] - 1)
            x[i, :, :, 0] = image_data[:, :, slice_index]
            y[i, :, :, 0] = mask_data[:, :, slice_index]
        
        return x, y


batch_size = 16 # The batch size to use when training the model
image_size = (224, 224)  # The size of the images


pip install tensorflow
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

def encoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = UpSampling2D(size=(2, 2))(input_tensor)
    x = Concatenate()([x, skip_tensor])
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_segnet_model(input_img, n_filters=64, n_classes=1, dropout=0.1, batchnorm=True):
    # Contracting Path (encoder)
    c1 = encoder_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = encoder_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = encoder_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = encoder_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Expanding Path (decoder)
    u6 = decoder_block(c4, c3, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    u7 = decoder_block(u6, c2, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    u8 = decoder_block(u7, c1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    # Output layer
    output_img = Conv2D(n_classes, (1, 1), activation='sigmoid')(u8)
    
    return Model(inputs=input_img, outputs=output_img)


from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)

def calculate_tpr_fpr(y_true, y_pred):
    # Assuming y_pred is sigmoid output, threshold to get binary mask
    y_pred = y_pred > 0.5
    # Flatten the arrays to compute confusion matrix
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
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

# ModelCheckpoint
checkpoint_path = "/Users/tongfeiyang/Desktop/liver_project/model_checkpoint.h5"
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)


from sklearn.model_selection import KFold

# Assuming image_paths and mask_paths are your full datasets
data_size = len(image_paths)
indices = np.arange(data_size)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []


# Assuming 'kf', 'image_paths', 'mask_paths', 'batch_size', 'image_size' are defined
for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
    print(f"Training on fold {fold+1}/5...")
    
    # Split your data
    train_images_subset = [image_paths[i] for i in train_indices]
    train_masks_subset = [mask_paths[i] for i in train_indices]
    val_images_subset = [image_paths[i] for i in val_indices]
    val_masks_subset = [mask_paths[i] for i in val_indices]
    
    # Data generators
    train_generator = NiiDataGenerator(train_images_subset, train_masks_subset, batch_size, image_size)
    val_generator = NiiDataGenerator(val_images_subset, val_masks_subset, batch_size, image_size)

    # Model setup and training
    input_img = Input((224, 224, 1), name='img')
    model = get_segnet_model(input_img)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef, iou_coef])
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=300, 
        callbacks=[model_checkpoint, early_stopping, lr_scheduler])

    # Predict on the validation set
    val_predictions = []
    val_trues = []
    for i in range(len(val_generator)):
        x, y = val_generator[i]
        preds = model.predict(x)
        val_predictions.extend(preds)
        val_trues.extend(y)
    
    val_predictions = np.array(val_predictions)
    val_trues = np.array(val_trues)

    # Calculate TPR and FPR
    tpr, fpr = calculate_tpr_fpr(val_trues, val_predictions)
    
    # Save fold results
    fold_results.append({
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_dice_coef': history.history['dice_coef'],
        'val_dice_coef': history.history['val_dice_coef'],
        'train_iou_coef': history.history['iou_coef'],
        'val_iou_coef': history.history['val_iou_coef'],
        'tpr': tpr,
        'fpr': fpr
    })

# Recalculating the average metrics across all folds
avg_metrics = {
    'train_loss': np.mean([fold['train_loss'] for fold in fold_results], axis=0),
    'val_loss': np.mean([fold['val_loss'] for fold in fold_results], axis=0),
    'train_dice_coef': np.mean([fold['train_dice_coef'] for fold in fold_results], axis=0),
    'val_dice_coef': np.mean([fold['val_dice_coef'] for fold in fold_results], axis=0),
    'train_iou_coef': np.mean([fold['train_iou_coef'] for fold in fold_results], axis=0),
    'val_iou_coef': np.mean([fold['val_iou_coef'] for fold in fold_results], axis=0),
    'tpr': np.mean([fold['tpr'] for fold in fold_results]),
    'fpr': np.mean([fold['fpr'] for fold in fold_results]),
}

# Plotting metrics for each fold and the averages
fig, axes = plt.subplots(5, 2, figsize=(15, 25))  # 5 metrics (loss, dice coef, iou coef, tpr, fpr), 2 columns (train, val)
metrics = ['loss', 'dice_coef', 'iou_coef']
titles = ['Loss', 'Dice Coefficient', 'IoU Coefficient', 'True Positive Rate', 'False Positive Rate']
epochs = range(1, 6)

# Plot for each metric
for i, metric in enumerate(metrics):
    for fold_idx, fold in enumerate(fold_results):
        axes[i, 0].plot(epochs, fold[f'train_{metric}'], label=f'Fold {fold_idx+1} Train')
        axes[i, 1].plot(epochs, fold[f'val_{metric}'], label=f'Fold {fold_idx+1} Val')
    # Averages
    axes[i, 0].plot(epochs, avg_metrics[f'train_{metric}'], label='Average Train', linewidth=2, color='black')
    axes[i, 1].plot(epochs, avg_metrics[f'val_{metric}'], label='Average Val', linewidth=2, color='black')
    axes[i, 0].set_title(f'Train {titles[i]}')
    axes[i, 1].set_title(f'Val {titles[i]}')
    axes[i, 0].legend()
    axes[i, 1].legend()

# True Positive Rate and False Positive Rate (only one value, not over epochs)
axes[3, 0].bar(np.arange(1, len(fold_results) + 1), [fold['tpr'] for fold in fold_results], color='skyblue', label='Fold TPR')
axes[3, 0].set_title('True Positive Rate per Fold')
axes[3, 1].bar(np.arange(1), [avg_metrics['tpr']], color='orange', label='Average TPR')
axes[3, 1].set_title('Average True Positive Rate')
axes[3, 0].set_xticks(np.arange(1, len(fold_results) + 1))
axes[3, 1].set_xticks([])

axes[4, 0].bar(np.arange(1, len(fold_results) + 1), [fold['fpr'] for fold in fold_results], color='lightgreen', label='Fold FPR')
axes[4, 0].set_title('False Positive Rate per Fold')
axes[4, 1].bar(np.arange(1), [avg_metrics['fpr']], color='red', label='Average FPR')
axes[4, 1].set_title('Average False Positive Rate')
axes[4, 0].set_xticks(np.arange(1, len(fold_results) + 1))
axes[4, 1].set_xticks([])

fig.tight_layout()
plt.show()

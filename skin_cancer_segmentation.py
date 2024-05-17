import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

# Load Data
# Load Data
def load_data(image_folder, mask_folder, img_size=(128, 128)):
    images = []
    masks = []
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.png')))  # .png uzantılı maskeleri al
    
    if len(image_files) != len(mask_files):
        raise ValueError("Number of image files does not match number of mask files")
    
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: {img_file} could not be loaded.")
            continue
        
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize
        images.append(img)
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: {mask_file} could not be loaded.")
            continue
        
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # Normalize
        masks.append(mask)
        
    return np.array(images), np.array(masks)


# Build U-Net Model
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_model(X_train, y_train, X_test, y_test):
    model = unet_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)
    
    # Save the model
    model.save('unet_skin_cancer_segmentation_model.h5')
    
    # Plot training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    return model

# Segment and Crop
def segment_and_crop(image_path, model):
    original_img = cv2.imread(image_path)
    img = cv2.resize(original_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    prediction = (prediction > 0.5).astype(np.uint8)[0, :, :, 0]
    
    prediction = cv2.resize(prediction, (original_img.shape[1], original_img.shape[0]))
    
    contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = original_img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return original_img, cropped_images

# Main script
if __name__ == "__main__":
    image_folder = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\ham1000-segmentation-and-classification\\images'
    mask_folder = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\ham1000-segmentation-and-classification\\masks'

    X, y = load_data(image_folder, mask_folder)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    
    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Example usage for segmentation and cropping
    test_image_path = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\ham1000-segmentation-and-classification\\masks\\ISIC_0024313.jpg'  # Adjust this path
    segmented_img, cropped_imgs = segment_and_crop(test_image_path, model)
    
    # Display the results
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    for i, cropped_img in enumerate(cropped_imgs):
        plt.figure()
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.show()

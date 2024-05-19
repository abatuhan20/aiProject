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

def load_data(image_folder, mask_folder, img_size=(128, 128)):
    images = []
    masks = []
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
    
    if len(image_files) != len(mask_files):
        raise ValueError("Number of image files does not match number of mask files")
    
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: {img_file} could not be loaded.")
            continue
        
        img = cv2.resize(img, img_size)
        img = img / 255.0
        images.append(img)
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: {mask_file} could not be loaded.")
            continue
        
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0
        masks.append(mask)
        
    return np.array(images), np.array(masks)


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


def train_model(X_train, y_train, X_test, y_test):
    model = unet_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)
    

    model.save('unet_skin_cancer_segmentation_model3.h5')
    

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    return model


def segment_and_crop(image_path, model):
    original_img = cv2.imread(image_path)
    img = cv2.resize(original_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img).squeeze()
    prediction = (prediction > 0.5).astype(np.uint8)
    
    prediction = cv2.resize(prediction, (original_img.shape[1], original_img.shape[0]))
    
    contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = original_img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return original_img, cropped_images


def visualize_predictions(X_test, y_test, model, num_images=5):
    for i in range(num_images):
        img = X_test[i]
        true_mask = y_test[i].squeeze()
        pred_mask = model.predict(np.expand_dims(img, axis=0)).squeeze()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        plt.title('True Mask')
        plt.imshow(true_mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask, cmap='gray')

        plt.show()

if __name__ == "__main__":
    image_folder = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\university-of-waterloo-skin-cancer-db-80-10-10\\train\\img'
    mask_folder = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\university-of-waterloo-skin-cancer-db-80-10-10\\train\\mask'

    X, y = load_data(image_folder, mask_folder)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    
    model = train_model(X_train, y_train, X_test, y_test)
    
    visualize_predictions(X_test, y_test, model)
    
    test_image_path = 'C:\\Users\\Arda\\Desktop\\Python\\yapayZekaProje\\university-of-waterloo-skin-cancer-db-80-10-10\\test\\img\\46_orig.jpg'
    segmented_img, cropped_imgs = segment_and_crop(test_image_path, model)
    
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    for i, cropped_img in enumerate(cropped_imgs):
        plt.figure()
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Cropped Image {i + 1}')
        plt.show()
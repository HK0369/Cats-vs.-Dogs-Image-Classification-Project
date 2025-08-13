# 4_model_evaluation.py
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

# ==== PATHS ====
BASE_DIR = r"C:\Users\nhari\OneDrive\Desktop\cats_vs_dogs_project"
TEST_DIR = os.path.join(BASE_DIR, "data", "processed", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cats_vs_dogs_model.h5")

# ==== LOAD MODEL ====
print(f"✅ Loading trained model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ==== DATA GENERATOR FOR TEST SET ====
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ==== EVALUATE MODEL ====
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Test Loss: {loss:.4f}")

# ==== OPTIONAL: PREDICT ON SINGLE IMAGE ====
def predict_single_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 150, 150, 3)
    
    prediction = model.predict(img_array)
    label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
    print(f"📌 Prediction for {os.path.basename(image_path)}: {label} ({prediction[0][0]:.4f})")
    return label

# Example usage:
# predict_single_image(r"C:\Users\nhari\OneDrive\Desktop\cats_vs_dogs_project\data\processed\test\cats\cat.4001.jpg")

# ==== OPTIONAL: PLOT PREDICTIONS ON A FEW TEST IMAGES ====
def plot_test_predictions(num_images=6):
    x, y = next(test_generator)  # get a batch
    preds = model.predict(x)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(x[i])
        true_label = 'Dog' if y[i]==1 else 'Cat'
        pred_label = 'Dog' if preds[i][0]>0.5 else 'Cat'
        plt.title(f"T:{true_label}\nP:{pred_label}")
        plt.axis('off')
    plt.show()

# Example usage:
# plot_test_predictions()

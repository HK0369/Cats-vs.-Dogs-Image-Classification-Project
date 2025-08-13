import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==== PATHS ====
# Using absolute paths for reliability
BASE_DIR = r"C:\Users\nhari\OneDrive\Desktop\cats_vs_dogs_project"
TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "processed", "val") # Using the correct validation set
MODEL_PATH = os.path.join(BASE_DIR, "models", "cats_vs_dogs_model.h5")

# ==== DATA CHECK ====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Please run the initial training first.")
if not os.listdir(TRAIN_DIR) or not os.listdir(VAL_DIR):
    raise FileNotFoundError("❌ Dataset folders are empty. Please run data preparation first.")

# ==== LOAD EXISTING MODEL ====
print(f"✅ Loading base model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ==== DATA AUGMENTATION ====
# This helps prevent overfitting by creating modified versions of your images
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,      # Increased rotation
    width_shift_range=0.25, # Increased shift
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,        # Increased zoom
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255) # No augmentation for validation data

# ==== DATA GENERATORS ====
print(f"📂 Loading training data from: {TRAIN_DIR}")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print(f"📂 Loading validation data from: {VAL_DIR}")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False # No need to shuffle validation data
)

# ==== COMPILE THE MODEL ====
# Re-compile the model to ensure the optimizer state is fresh for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==== ADVANCED TRAINING CALLBACKS ====
# 1. ModelCheckpoint: Saves the model only when `val_accuracy` improves.
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_PATH,          # Save the best model to the same path
    save_best_only=True,          # Only save if the model is better
    monitor='val_accuracy',       # The metric to monitor
    mode='max',                   # We want to maximize accuracy
    verbose=1                     # Print a message when saving
)

# 2. EarlyStopping: Stops training if `val_accuracy` doesn't improve for 5 epochs.
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=5,                   # Number of epochs to wait for improvement
    verbose=1,
    restore_best_weights=True     # Restore model weights from the best epoch
)


# ==== TRAIN THE MODEL FOR MORE EPOCHS ====
print("\n🚀 Starting advanced training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=25,                    # Increased number of epochs
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_callback, early_stopping_callback] # Add the new callbacks
)

print(f"\n✅ Training complete. Best model saved at: {MODEL_PATH}")

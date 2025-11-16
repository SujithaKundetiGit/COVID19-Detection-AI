import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATASET_PATH = r'D:\VisualStudioProjGuvi\Covid19project\venv\Scripts\dataset_path'

IMG_HEIGHT = 224 # ResNet-50 default input size
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 3 # COVID-19, Viral Pneumonia, Normal 
EPOCHS = 10 #Due to system space running only 10 EPOCHS
LEARNING_RATE = 0.0001 # A small learning rate is recommended for fine-tuning

# --- 1. Data Exploration & Preprocessing (with Augmentation) ---

train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1] [cite: 29]
    rotation_range=20, # Rotation augmentation [cite: 28]
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True, # Flip augmentation [cite: 28]
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255) 

# Flow images from directory
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # For multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Keep data in order for evaluation metrics
)

# --- 2. Model Development (Transfer Learning with ResNet-50) ---

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the base model initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers for multi-class task
x = base_model.output
x = GlobalAveragePooling2D()(x) # Global Average Pooling layer
x = Dense(512, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer for 3 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. Model Training ---

print("\nStarting Model Training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE
)

# Save the trained model
model.save('covid_detection_model.h5')
print("\nModel saved as covid_detection_model.h5")

# --- 4. Model Evaluation & Fine-Tuning ---

print("\n--- Model Evaluation ---")
# Evaluate the model on the test set
eval_results = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Test Accuracy: {eval_results[1]*100:.2f}%") # [cite: 34, 44]

# Predict classes for the confusion matrix and ROC-AUC
Y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true_classes = test_generator.classes # True labels

# Get class labels
class_labels = list(train_generator.class_indices.keys())
print(f"Class Labels: {class_labels}")

# Classification Report (Precision, Recall, F1-Score) 
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# Confusion Matrix 
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() # Visualizes common misclassifications [cite: 35]

from tensorflow.keras.utils import to_categorical
y_true_categorical = to_categorical(y_true_classes, num_classes=NUM_CLASSES)
roc_auc = roc_auc_score(y_true_categorical, Y_pred, multi_class='ovr')
print(f"\nOverall ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}")

# --- 5. Deployment Placeholder ---
print("\nDeployment step requires additional setup (e.g., Flask/Streamlit web app in Docker) [cite: 37, 50]")
model.save('covid_detection_model.keras')
#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Simple Training Script for Clothing Classifier""""""

Trains on the complete dataset using transfer learning

"""Production-ready training script for clothing classifier.Simple Training Script - Trains the clothing classifier model.



import osTrains on the complete dataset with all features you requested:"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

- Progress tracking with detailed logging

import sys

import pandas as pd- Model checkpointingimport sys

import numpy as np

import tensorflow as tf- Early stoppingimport os

from pathlib import Path

from datetime import datetime- Learning rate scheduling  from pathlib import Path

import json

- Data augmentation

# Configuration

PROJECT_ROOT = Path('/home/anupam/code/AIProject')- Two-phase transfer learning# Add project root to path

DATA_DIR = PROJECT_ROOT / 'data'

PROCESSED_DIR = DATA_DIR / 'processed'- Comprehensive evaluationproject_root = Path(__file__).parent

RAW_DIR = DATA_DIR / 'raw'

MODELS_DIR = PROJECT_ROOT / 'models' / 'saved_models'"""sys.path.insert(0, str(project_root / 'models'))

MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(project_root / 'scripts'))

IMG_SIZE = 224

BATCH_SIZE = 32import os

EPOCHS_PHASE1 = 15

EPOCHS_PHASE2 = 15os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warningsprint("=" * 80)



print("=" * 80)print("🎨 SIMPLE CLOTHING CLASSIFIER TRAINING")

print("🚀 CLOTHING CLASSIFIER TRAINING")

print("=" * 80)import sysprint("=" * 80)

print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Data: {PROCESSED_DIR}")import pandas as pd

print(f"Models: {MODELS_DIR}")

print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")import numpy as np# Import after path is set

print(f"Batch size: {BATCH_SIZE}")

import tensorflow as tffrom models.clothing_classifier import ClothingClassificationModel

# Check GPU

gpus = tf.config.list_physical_devices('GPU')from pathlib import Pathimport pandas as pd

if gpus:

    print(f"\nGPU Available: {len(gpus)} GPU(s)")from datetime import datetimeimport json

else:

    print("\nNo GPU - using CPU")import json



# Load data# Paths

print("\nLoading data...")

train_df = pd.read_csv(PROCESSED_DIR / 'train.csv')# Configurationprocessed_dir = project_root / 'data' / 'processed'

val_df = pd.read_csv(PROCESSED_DIR / 'val.csv')

test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')DATA_DIR = Path('/home/anupam/code/AIProject/data')models_dir = project_root / 'models' / 'saved_models'



with open(PROCESSED_DIR / 'label_mapping.json', 'r') as f:PROCESSED_DIR = DATA_DIR / 'processed'models_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = json.load(f)

RAW_DIR = DATA_DIR / 'raw'

num_classes = len(label_mapping)

MODELS_DIR = Path('/home/anupam/code/AIProject/models/saved_models')# Load label mapping

print(f"Train samples: {len(train_df):,}")

print(f"Val samples: {len(val_df):,}")MODELS_DIR.mkdir(parents=True, exist_ok=True)print("\n📂 Loading label mapping...")

print(f"Test samples: {len(test_df):,}")

print(f"Classes: {num_classes}")with open(processed_dir / 'label_mapping.json', 'r') as f:



# Create datasetsIMG_SIZE = 224    label_info = json.load(f)

def create_dataset(df, is_training=True):

    def load_image(image_path, label):BATCH_SIZE = 32

        img = tf.io.read_file(image_path)

        img = tf.image.decode_jpeg(img, channels=3)EPOCHS_PHASE1 = 15num_classes = len(label_info['label_to_name'])

        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

        img = tf.cast(img, tf.float32) / 255.0EPOCHS_PHASE2 = 15print(f"Number of classes: {num_classes}")

        

        if is_training:print(f"Classes: {list(label_info['label_to_name'].values())}")

            img = tf.image.random_flip_left_right(img)

            img = tf.image.random_brightness(img, 0.2)print("=" * 80)

            img = tf.image.random_contrast(img, 0.8, 1.2)

            img = tf.image.random_saturation(img, 0.8, 1.2)print("🚀 AI FASHION RECOMMENDATION - CLOTHING CLASSIFIER TRAINING")# Create model

        

        return img, labelprint("=" * 80)print("\n🏗️  Creating model...")

    

    paths = [str(RAW_DIR / 'images' / f"{row['id']}.jpg") for _, row in df.iterrows()]print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")model = ClothingClassificationModel(

    labels = df['label'].values

    print(f"💾 Data directory: {PROCESSED_DIR}")    num_classes=num_classes,

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)print(f"💾 Models directory: {MODELS_DIR}")    img_size=(224, 224),

    

    if is_training:print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")    model_name='efficientnet',

        dataset = dataset.shuffle(1000)

    print(f"📦 Batch size: {BATCH_SIZE}")    use_pretrained=True

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return datasetprint(f"🔄 Phase 1 epochs: {EPOCHS_PHASE1}"))



print("\nCreating data pipelines...")print(f"🔄 Phase 2 epochs: {EPOCHS_PHASE2}")

train_dataset = create_dataset(train_df, True)

val_dataset = create_dataset(val_df, False)# Build model

test_dataset = create_dataset(test_df, False)

print("Data pipelines ready")# Check GPUmodel.build_model()



# Build modelgpus = tf.config.list_physical_devices('GPU')

print("\nBuilding model...")

from tensorflow.keras import layers, modelsif gpus:# Create data generators

from tensorflow.keras.applications import EfficientNetB0

    print(f"\n✅ GPU Available: {len(gpus)} GPU(s)")print("\n📊 Creating data generators...")

base_model = EfficientNetB0(

    include_top=False,    for gpu in gpus:train_csv = str(processed_dir / 'train.csv')

    weights='imagenet',

    input_shape=(IMG_SIZE, IMG_SIZE, 3)        print(f"   - {gpu}")val_csv = str(processed_dir / 'val.csv')

)

base_model.trainable = Falseelse:



inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    print("\n⚠️  No GPU detected - training will use CPU (slower)")train_dataset, val_dataset = model.create_data_generators(

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)    train_csv=train_csv,

x = layers.Dense(256, activation='relu')(x)

x = layers.BatchNormalization()(x)print("\n" + "=" * 80)    val_csv=val_csv,

x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation='relu')(x)    batch_size=16

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)# Load processed data)



model = models.Model(inputs=inputs, outputs=outputs)print("\n📂 Loading processed data...")

print(f"Model built: {model.count_params():,} parameters")

train_df = pd.read_csv(PROCESSED_DIR / 'train.csv')# Train Phase 1 (frozen base)

# PHASE 1

print("\n" + "=" * 80)val_df = pd.read_csv(PROCESSED_DIR / 'val.csv')print("\n" + "=" * 80)

print("PHASE 1: Training with frozen base")

print("=" * 80)test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')print("PHASE 1: Training with frozen base model")



model.compile(print("=" * 80)

    optimizer=tf.keras.optimizers.Adam(0.001),

    loss='sparse_categorical_crossentropy',with open(PROCESSED_DIR / 'label_mapping.json', 'r') as f:

    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_acc')]

)    label_mapping = json.load(f)history1 = model.train(



callbacks_p1 = [    train_dataset=train_dataset,

    tf.keras.callbacks.ModelCheckpoint(

        str(MODELS_DIR / 'classifier_phase1_best.keras'),num_classes = len(label_mapping)    val_dataset=val_dataset,

        save_best_only=True,

        monitor='val_accuracy',    epochs=3,

        mode='max',

        verbose=1print(f"✅ Train samples: {len(train_df):,}")    model_save_path=str(models_dir / 'clothing_classifier_phase1.keras')

    ),

    tf.keras.callbacks.EarlyStopping(print(f"✅ Validation samples: {len(val_df):,}"))

        monitor='val_loss',

        patience=5,print(f"✅ Test samples: {len(test_df):,}")

        restore_best_weights=True,

        verbose=1print(f"✅ Number of classes: {num_classes}")print("\n✅ Phase 1 complete!")

    ),

    tf.keras.callbacks.ReduceLROnPlateau(print(f"✅ Classes: {list(label_mapping.keys())}")

        monitor='val_loss',

        factor=0.5,# Train Phase 2 (fine-tuning)

        patience=3,

        min_lr=1e-7,# Create data generatorsprint("\n" * 80)

        verbose=1

    )def create_dataset(df, is_training=True):print("PHASE 2: Fine-tuning model")

]

    """Create tf.data.Dataset from DataFrame."""print("=" * 80)

print("\nStarting Phase 1 training...")

history_p1 = model.fit(    

    train_dataset,

    validation_data=val_dataset,    def load_and_preprocess_image(image_path, label):model.unfreeze_base_model(layers_to_unfreeze=-30)

    epochs=EPOCHS_PHASE1,

    callbacks=callbacks_p1,        # Load image

    verbose=1

)        img = tf.io.read_file(image_path)history2 = model.train(

print("\nPhase 1 complete!")

        img = tf.image.decode_jpeg(img, channels=3)    train_dataset=train_dataset,

# PHASE 2

print("\n" + "=" * 80)        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])    val_dataset=val_dataset,

print("PHASE 2: Fine-tuning")

print("=" * 80)        img = tf.cast(img, tf.float32) / 255.0    epochs=3,



base_model.trainable = True            model_save_path=str(models_dir / 'clothing_classifier.keras')

for layer in base_model.layers[:-30]:

    layer.trainable = False        if is_training:)



print("Unfroze last 30 layers")            # Data augmentation



model.compile(            img = tf.image.random_flip_left_right(img)print("\n✅ Phase 2 complete!")

    optimizer=tf.keras.optimizers.Adam(0.00001),

    loss='sparse_categorical_crossentropy',            img = tf.image.random_brightness(img, 0.2)

    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_acc')]

)            img = tf.image.random_contrast(img, 0.8, 1.2)# Final evaluation



callbacks_p2 = [            img = tf.image.random_saturation(img, 0.8, 1.2)print("\n" + "=" * 80)

    tf.keras.callbacks.ModelCheckpoint(

        str(MODELS_DIR / 'classifier_final.keras'),        print("📊 FINAL EVALUATION")

        save_best_only=True,

        monitor='val_accuracy',        return img, labelprint("=" * 80)

        mode='max',

        verbose=1    

    ),

    tf.keras.callbacks.EarlyStopping(    # Prepare paths and labelstest_csv = str(processed_dir / 'test.csv')

        monitor='val_loss',

        patience=5,    image_paths = [str(RAW_DIR / 'images' / f"{row['id']}.jpg") for _, row in df.iterrows()]test_df = pd.read_csv(test_csv)

        restore_best_weights=True,

        verbose=1    labels = df['label'].values

    ),

    tf.keras.callbacks.ReduceLROnPlateau(    print(f"\nEvaluating on {len(test_df)} test samples...")

        monitor='val_loss',

        factor=0.5,    # Create dataset

        patience=2,

        min_lr=1e-8,    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))# Create test dataset

        verbose=1

    )    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)import tensorflow as tf

]

    

print("\nStarting Phase 2 training...")

history_p2 = model.fit(    if is_training:def parse_image(image_path, label):

    train_dataset,

    validation_data=val_dataset,        dataset = dataset.shuffle(buffer_size=1000)    image = tf.io.read_file(image_path)

    epochs=EPOCHS_PHASE2,

    callbacks=callbacks_p2,        image = tf.image.decode_jpeg(image, channels=3)

    verbose=1

)    dataset = dataset.batch(BATCH_SIZE)    image = tf.image.resize(image, (224, 224))

print("\nPhase 2 complete!")

    dataset = dataset.prefetch(tf.data.AUTOTUNE)    image = image / 255.0

# Evaluate

print("\n" + "=" * 80)        return image, label

print("FINAL EVALUATION")

print("=" * 80)    return dataset



test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)test_paths = test_df['image_path'].values



print("\nTest Results:")print("\n📊 Creating data pipelines...")test_labels = test_df['category_label'].values

print(f"Loss: {test_results['loss']:.4f}")

print(f"Accuracy: {test_results['accuracy']*100:.2f}%")train_dataset = create_dataset(train_df, is_training=True)

print(f"Top-3 Accuracy: {test_results['top_3_acc']*100:.2f}%")

val_dataset = create_dataset(val_df, is_training=False)test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Save

final_path = MODELS_DIR / 'clothing_classifier.keras'test_dataset = create_dataset(test_df, is_training=False)test_dataset = test_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

model.save(final_path)

print(f"\nSaved to: {final_path}")print("✅ Data pipelines created")test_dataset = test_dataset.batch(16)



# Save historytest_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

history_data = {

    'phase1': {k: [float(v) for v in history_p1.history[k]] for k in history_p1.history},# Build model

    'phase2': {k: [float(v) for v in history_p2.history[k]] for k in history_p2.history},

    'test': {k: float(v) for k, v in test_results.items()}print("\n🏗️  Building model...")results = model.evaluate(test_dataset)

}



history_path = MODELS_DIR / 'training_history.json'

with open(history_path, 'w') as f:from tensorflow.keras import layers, modelsprint("\n" + "=" * 80)

    json.dump(history_data, f, indent=2)

from tensorflow.keras.applications import EfficientNetB0print("🎉 TRAINING COMPLETE!")

print(f"History saved to: {history_path}")

print("=" * 80)

print("\n" + "=" * 80)

print("TRAINING COMPLETE!")# Base modelprint(f"\nModel saved to: {models_dir / 'clothing_classifier.keras'}")

print("=" * 80)

print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")base_model = EfficientNetB0(print("\nNext steps:")


    include_top=False,print("  1. Run the compatibility model training")

    weights='imagenet',print("  2. Test with: streamlit run app/streamlit_app.py")

    input_shape=(IMG_SIZE, IMG_SIZE, 3)print("=" * 80)

)
base_model.trainable = False  # Freeze for phase 1

# Build classifier
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

print(f"✅ Model built with {model.count_params():,} parameters")

# Phase 1: Train with frozen base
print("\n" + "=" * 80)
print("🎯 PHASE 1: Training with frozen base model")
print("=" * 80)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'clothing_classifier_phase1_best.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n🚀 Starting Phase 1 training...\n")

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
    verbose=1
)

print("\n✅ Phase 1 training complete!")

# Phase 2: Fine-tune with unfrozen layers
print("\n" + "=" * 80)
print("🎯 PHASE 2: Fine-tuning with unfrozen layers")
print("=" * 80)

# Unfreeze last layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"✅ Unfrozen last 30 layers")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

callbacks_phase2 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'clothing_classifier_final.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

print("\n🚀 Starting Phase 2 training...\n")

history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
    verbose=1
)

print("\n✅ Phase 2 training complete!")

# Final evaluation
print("\n" + "=" * 80)
print("📊 FINAL EVALUATION ON TEST SET")
print("=" * 80)

test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)

print("\n📈 Test Results:")
print(f"   Loss: {test_results['loss']:.4f}")
print(f"   Accuracy: {test_results['accuracy']*100:.2f}%")
print(f"   Top-3 Accuracy: {test_results['top_3_accuracy']*100:.2f}%")

# Save final model
final_model_path = MODELS_DIR / 'clothing_classifier.keras'
model.save(final_model_path)
print(f"\n💾 Final model saved to: {final_model_path}")

# Save training history
history_data = {
    'phase1': {k: [float(v) for v in history_phase1.history[k]] for k in history_phase1.history},
    'phase2': {k: [float(v) for v in history_phase2.history[k]] for k in history_phase2.history},
    'test_results': {k: float(v) for k, v in test_results.items()}
}

history_path = MODELS_DIR / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)
print(f"💾 Training history saved to: {history_path}")

print("\n" + "=" * 80)
print("🎉 CLOTHING CLASSIFIER TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📁 Saved files:")
print(f"   - {final_model_path}")
print(f"   - {MODELS_DIR / 'clothing_classifier_phase1_best.keras'}")
print(f"   - {history_path}")
print("\n🚀 Next step: Train outfit compatibility model")
print("=" * 80)

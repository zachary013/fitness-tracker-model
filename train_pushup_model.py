import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
from datetime import datetime
import time


class DataCollector:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.up_dir = os.path.join(output_dir, "up")
        self.down_dir = os.path.join(output_dir, "down")

        # Create directories if they don't exist
        os.makedirs(self.up_dir, exist_ok=True)
        os.makedirs(self.down_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0

    def collect_data(self):
        print("Data Collection Started")
        print("Press 'u' to capture UP position")
        print("Press 'd' to capture DOWN position")
        print("Press 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Display frame count
            cv2.putText(frame, f'Frames: {self.frame_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('u'):
                # Save up position
                self._save_frame(frame, "up")
            elif key == ord('d'):
                # Save down position
                self._save_frame(frame, "down")
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _save_frame(self, frame, position):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = self.up_dir if position == "up" else self.down_dir
        filename = os.path.join(output_dir, f"{position}_{timestamp}.jpg")

        # Resize frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(filename, frame)
        self.frame_count += 1
        print(f"Saved {position} position frame: {filename}")


class DataPreprocessor:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.image_size = (224, 224)
        self.batch_size = 32

        # Create data generator with augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

    def get_generators(self):
        train_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, validation_generator


class ImprovedPushupModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        # Load the MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Fine-tune the last few layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Add custom top layers with improved architecture
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # First dense block
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(0.5)(x)

        # Second dense block
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(0.3)(x)

        # Output layer
        predictions = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile with better learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, train_generator, validation_generator, epochs=50):
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_pushup_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def save_model(self, filename='pushup_model.h5'):
        self.model.save(filename)


def main():
    # 1. Collect Data
    print("Starting data collection...")
    collector = DataCollector()
    collector.collect_data()

    # 2. Preprocess Data
    print("\nPreparing data generators...")
    preprocessor = DataPreprocessor()
    train_generator, validation_generator = preprocessor.get_generators()

    # 3. Train Model
    print("\nTraining model...")
    model = ImprovedPushupModel()
    history = model.train(train_generator, validation_generator)

    # 4. Save Model
    model.save_model()
    print("\nTraining completed! Model saved as 'pushup_model.h5'")


if __name__ == "__main__":
    main()
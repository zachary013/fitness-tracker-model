import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


class MultiExerciseModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        # Load the MobileNetV2 model with pre-trained weights
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze most layers for transfer learning, except the last few
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Dense block 1
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Dense block 2
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Output layer for multi-class classification
        predictions = Dense(5, activation='softmax')(x)  # 5 classes for the exercises

        # Create and compile the model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_generator, validation_generator, epochs=50):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Modified callbacks with proper file path
        model_path = os.path.join('models', 'exercise_model.keras')
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def save_model(self, filename='exercise_model.keras'):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save model in the models directory
        model_path = os.path.join('models', filename)
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")


class DataPreprocessor:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.image_size = (224, 224)
        self.batch_size = 32

        # Data augmentation and preprocessing
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% for validation
        )

    def get_generators(self):
        # Training data generator
        train_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Validation data generator
        validation_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, validation_generator

def main():
    # Enable memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Step 1: Preprocess the data
    print("Preparing data generators...")
    preprocessor = DataPreprocessor()
    train_generator, validation_generator = preprocessor.get_generators()

    # Step 2: Initialize and train the model
    print("\nTraining the model...")
    model = MultiExerciseModel()
    history = model.train(train_generator, validation_generator, epochs=50)

    # Step 3: Save the trained model
    model.save_model()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
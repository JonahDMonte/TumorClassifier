import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directory and batch size
data_dir = r'/content/Data'  # Replace with the actual path to your data directory
batch_size = 32  # Adjust this according to your needs
image_height, image_width = 256, 256  # Adjust to match your model's input size

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values to [0, 1]
    validation_split=0.2  # Split data into training and validation sets
)

# Create data generators for training and validation sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),  # Specify the input size of your model
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'  # Specify 'training' or 'validation'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create and compile your model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Adjust the number of output classes
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train your model
epochs = 10  # Adjust the number of training epochs as needed
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)
model.save("tf_cnn_2")
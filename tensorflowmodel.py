import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# Define data directory and batch size
data_dir = r'C:\Users\Jonah\PycharmProjects\TumorClassifier\Data'  # Replace with the actual path to your data directory
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
    tf.keras.layers.Input(shape=(256,256,3)),
    hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b7/classification/1'),
    #the layer which loads eff_net b7
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train your model
epochs = 20  # Adjust the number of training epochs as needed
with tf.device('/GPU:0'):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
model.save("tf_efficientnet")
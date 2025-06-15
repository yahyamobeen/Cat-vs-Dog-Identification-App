from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(img_size=(128, 128), batch_size=32):
    # Data generators for training and test
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_data = train_datagen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_data = test_datagen.flow_from_directory(
        'data/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_data, test_data

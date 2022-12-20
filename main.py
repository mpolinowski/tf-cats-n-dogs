import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


# # eval the un-modified model
# # import resnet50 with imagenet weights + include the top layers
# model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = True)
# # resnet50 expects image to be of shape (1, 224, 224, 3)
# sample_image= tf.keras.preprocessing.image.load_img(r'./test_images/cat.png', target_size = (224, 224))
# sample_image = np.expand_dims(sample_image, axis = 0)
# print('image shape: ',np.shape(sample_image))
# # keras offers resnet50 preprocess preset we can use
# # image will be processed identically to training images
# preprocessed_image = tf.keras.applications.resnet50.preprocess_input(sample_image)
# # run prediction
# predictions = model.predict(preprocessed_image)
# # use keras resnet50 prediction decoder to return top5 predictions
# print('predictions:', tf.keras.applications.resnet50.decode_predictions(predictions, top = 5)[0])


# load only the convolution layers / general feature detection of resnet50
base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False)

# take base model convolution layers from resnet
x = base_model.output
# compress incoming feature maps from resnet layers
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# and add fresh top of dense layers
# each node will distinguish between 1024 or 512 features
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dense(512, activation = 'relu')(x)
# the final layer breaks everything down to a binary decision - cat or dog
preds = tf.keras.layers.Dense(2, activation = 'softmax')(x)
# create new model
model = tf.keras.models.Model(inputs = base_model.input, outputs = preds)

# lock all resnet layers 1-174
for layer in model.layers[:175]:
    layer.trainable = False
# the new dense layers have to be trainable
for layer in model.layers[175:]:
    layer.trainable = True

# tell keras to use resnet preset
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function= tf.keras.applications.resnet50.preprocess_input)
# and define the training dataset
train_generator = train_datagen.flow_from_directory('./data/training_set/', 
                                                   target_size = (224, 224),
                                                   color_mode = 'rgb',
                                                   batch_size = 32,
                                                   class_mode = 'categorical',
                                                   shuffle = True)

# compile the new model
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# and train it on your dataset
history = model.fit(train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs = 5)


# evaluating the model - accuracy & loss
acc = history.history['accuracy']
loss = history.history['loss']
## plot accuracy
plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
## plot loss
plt.figure()
plt.plot(loss, label='Training Loss')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()


# take a sample image for testing
Sample_Image= tf.keras.preprocessing.image.load_img(r'./test_images/cat.png', target_size = (224, 224))
plt.imshow(Sample_Image)
plt.show()
## pre-process for resnet
Sample_Image = tf.keras.preprocessing.image.img_to_array(Sample_Image)
np.shape(Sample_Image)
Sample_Image = np.expand_dims(Sample_Image, axis = 0)
## run prediction
Sample_Image = tf.keras.applications.resnet50.preprocess_input(Sample_Image)
predictions = model.predict(Sample_Image)
print('Predictions:', predictions)
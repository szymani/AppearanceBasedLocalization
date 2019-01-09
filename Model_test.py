from keras.models import model_from_json
import os
import random
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt


dataset_path = 'Sekcje'
data_dict = []
all_dict = []
image_height = 120
image_width = 160

# Model reconstruction from JSON file
with open('models/siamese_net_lr10e-4.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('models/siamese_net_lr10e-4.h5')
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train path to sections !
for section in os.listdir(dataset_path):
    section_path = os.path.join(dataset_path, section)
    data_dict = os.listdir(section_path)
    for each in data_dict:
        each = os.path.join(section_path, each)
        all_dict.append(each)



results = [[]]
for i in range(0, 100):
    train_indexes = random.sample(range(0, len(all_dict) - 1), 2)

    print(train_indexes)
    print(all_dict[train_indexes[0]])
    print(all_dict[train_indexes[1]])

    pairs_of_images = [np.zeros((1, image_height, image_width, 1)) for i in range(2)]

    image = Image.open(all_dict[train_indexes[0]])
    image = np.asarray(image).astype(np.float64)
    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]  # my
    image = image / 255.0  # my
    pairs_of_images[0][0, :, :, 0] = image

    cv.imshow("First", image)

    image = Image.open(all_dict[train_indexes[1]])
    image = np.asarray(image).astype(np.float64)
    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image = image / 255.0
    pairs_of_images[1][0, :, :, 0] = image

    cv.imshow("Second", image)

    prob = model.predict(pairs_of_images)
    print(prob[0][0])
    if(all_dict[train_indexes[0]][-12:-9:1] == all_dict[train_indexes[1]][-12:-9:1]):
        results.append([1, prob[0][0]])
    else:
        results.append([0, prob[0][0]])
    cv.waitKey(3000)


results.pop(0)
plt.plot(results)
plt.show()




from PIL import Image
import random
import os
import numpy as np

data_images = []
for filename in os.listdir("data/images"):
    im = Image.open(os.path.join("data/images", filename))
    data_images.append(im)

dim = 50
numb_samples = 100
samples = []

for image in data_images:
    x, y = image.size

    for _ in range(numb_samples):
        x1 = random.randrange(0, x - dim)
        y1 = random.randrange(0, y - dim)

        samples.append(im.crop((x1, y1, x1 + dim, y1 + dim)))

to_remove = set(random.sample(range(len(samples)), 300))
# 1000 samples
samples = [x for i, x in enumerate(samples) if not i in to_remove]

samples[0].show()

# 1 d arrays
flat_arrays = []
# salt and pepper added
saltpep_arrays = []
for image in samples:
    image_arr = np.array(image)
    flat_arr = image_arr.flatten()
    for i in range(len(flat_arr)):
        proba = random.randrange(1,100)
        if proba <= 20:
            flip = random.randint(0, 1)
            if flip == 0:
                flat_arr[i] = 0
            else:
                flat_arr[i] = 255

normal_im = Image.fromarray(flat_arrays[3].reshape(50,50))
normal_im.show()

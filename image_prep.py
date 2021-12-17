from PIL import Image
import random
import os

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
samples = [x for i, x in enumerate(samples) if not i in to_remove]

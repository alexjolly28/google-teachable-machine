import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# image_sh = cv2.imread('test1.jpg') 

# Replace this with the path to your image
image = Image.open('test1.jpg')
image.show()

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
# image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array
labels=['dcmotor','arduino','none']
acus=[]
# run the inference
prediction = model.predict(data)
# print(prediction)
for accs in prediction:

    for acc in accs :
        acus.append(acc)
# print(acus)
zipped=dict(zip(labels, acus))
print(zipped)
for key,value in zipped.items():
    # print(keys,values)
    if value > 0.6:
        print(key)
        if key =='dcmotor':
            os.system('open Rotor_Complete.SLDPRT -a eDrawings')
        elif key =='arduino':
            os.system('open arduino\ uno.IGS -a eDrawings')
        else :
            print('none')
        # zipped=list(zip(labels,acc))
    # print(type(accs))
        # if acc >.6:
        #     print(acc)
    # print(acc)

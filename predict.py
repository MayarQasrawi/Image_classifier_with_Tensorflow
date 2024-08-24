from PIL import Image
import json
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import numpy as np





def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy().squeeze()
    return image


def predict(image_path, model, top_k, class_names):

    image = Image.open(image_path)

    image = np.asarray(image)
   
    processed_image = process_image(image)
    image = np.expand_dims(processed_image, axis = 0)



    ps = model.predict(image)[0]

    probabilities = np.sort(ps)[-top_k:len(ps)]
    prbabilities = probabilities.tolist()
    print(prbabilities)
    classes = np.argpartition(ps, -top_k)[-top_k:]
    classes = classes.tolist()
    names = [class_names.get(str(i )).capitalize() for i in (classes)] 
    print(names)


    class_names = [class_names.get(str(i)).capitalize() for i in classes]
    ps_cl = list(zip(prbabilities, names))
    print(ps_cl)
    return probabilities, names

#intialize the batch size and image size
batch_size = 32
image_size = 224

pars = argparse.ArgumentParser()
pars.add_argument('image_path')
pars.add_argument('model')
pars.add_argument('--top_k')
pars.add_argument('--class_file') 


'''
 Using parse_args() can make your code more readable
 by separating the argument parsing logic from the rest of your code.
'''
args = parser.parse_args()
    print(args)
    print('arg1:', args.image_path)
    print('arg2:', args.model)
    print('top_k:', args.top_k)
    print('class_file:', args.class_file)


    

top_k = args.top_k
if top_k is None: 
    top_k = 5
model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
class_names=[]
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

image_path = args.image_path

probs, classes = predict(image_path, model, top_k,class_names)

print('classes:',classes)
print('proediction:', probs)

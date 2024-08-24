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




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('model_path', help='Path to the TensorFlow model')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--category_names', help='Path to the JSON file containing category names')
    return parser.parse_args()

def load_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()

    model = load_model(args.model_path)
    class_names = load_category_names(args.category_names)
    image_path=args.image_path
    top_k=args.top_k

    predictions, classes = predict(image_path, model, top_k,class_names)
    print('classes:',classes)
    print('prediction for each class:',predictions)
if __name__ == '__main__':
    main()

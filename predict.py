import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def process_image(np_image, img_size=224):
    img = tf.convert_to_tensor(np_image)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def predict(image_path, model_path, top_k=5):
    # uplod pic
    im = Image.open(image_path)
    np_im = np.asarray(im)

    # preprocessing + batch dim
    proc = process_image(np_im)
    proc = np.expand_dims(proc, axis=0)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    # predect
    preds = model.predict(proc, verbose=0)[0]

    top_idx = preds.argsort()[-top_k:][::-1]
    top_probs = preds[top_idx]
    top_classes = [str(i) for i in top_idx]
    return top_probs.tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description="Flower classifier")
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("model_path", type=str, help="Path to saved Keras model (.h5)")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K classes")
    parser.add_argument("--category_names", type=str, default=None,
                        help="Path to JSON mapping of labels to names (e.g., label_map.json)")
    args = parser.parse_args()

    probs, classes = predict(args.image_path, args.model_path, args.top_k)

    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_map = json.load(f)
        names = [cat_map.get(c, c) for c in classes]
    else:
        names = classes

    # print results
    print("Top probabilities:", probs)
    print("Top classes:", classes)
    print("Top names:", names)

if __name__ == "__main__":
    main()








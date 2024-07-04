# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 01:28:01 2022
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from PIL import Image
import os
import argparse
from tqdm import tqdm

# Suppress TensorFlow warnings and info messages
# Suppress TensorFlow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = [tf.math.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

def resize_image(fp, T):

    img = Image.open(fp)
    w, h = img.size
    
    if w> T or h > T:

        if w > h:
            new_w = T
            new_h = int((T / w) * h)
        else:
            new_h = T
            new_w = int((T / h) * w)
        
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return img

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.int32),
                                  tf.TensorSpec(shape=[], dtype=tf.float32)))
    def __call__(self, img, steps, step_size):
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
        return loss, img

def run_deep_dream(dream_model, img, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    while steps_remaining:
        steps_run = tf.constant(min(steps_remaining, 100))
        steps_remaining -= steps_run
        loss, img = deepdream(img, steps=steps_run, step_size=step_size)
    return deprocess(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepDream Script')
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--model_architecture', type=str, default='InceptionV3')
    parser.add_argument('--weights', type=str, default='imagenet')
    parser.add_argument('--max_dim', type=int, default=500, help='Maximum dimension for the downloaded image')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps for DeepDream')
    parser.add_argument('--step_size', type=float, default=0.01, help='Step size for DeepDream')
    args = parser.parse_args()

    architecture = getattr(tf.keras.applications, args.model_architecture)

    base_model = architecture(include_top=False, weights=args.weights)
    layers = [base_model.get_layer(name).output for name in ['mixed3', 'mixed5']]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    global deepdream
    deepdream = DeepDream(dream_model)

    fns = os.listdir(args.img_dir)
    fps = [os.path.join(args.img_dir,fn) for fn in fns]

    for fp in tqdm(fps, desc='Performing Deep Dream on present images'):

      arr = np.asarray(resize_image(fp, args.max_dim))
      dream_img = run_deep_dream(dream_model, img=arr, steps=args.steps, step_size=args.step_size).numpy()

      plt.imshow(dream_img)
      plt.axis('off')
      plt.savefig(fp.split('.')[0]+'_DD_hallucination.png')

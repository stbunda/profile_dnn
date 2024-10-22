import os

import tensorflow as tf
import torch
from torchvision import transforms

import time
import numpy as np
import psutil
import pandas
from tensorboard.plugins.text.text_plugin import process_event

from datasets.data import load_images_from_directory, batch_generator

# Disables eager execution
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


class CPU_Usage_Callback(object):
    def on_start(self):
        self.process = psutil.Process()
        self.cpu_start = self.process.cpu_percent()
        self.start = time.time()

    def on_end(self):
        self.end = time.time()
        self.cpu_end = self.process.cpu_percent()
        self.latency = self.end - self.start
        self.cpu_usage = self.cpu_end - self.cpu_start


class Throughput_Callback(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def on_start(self):
        self.start = time.time()

    def on_end(self, batch_idx):
        if batch_idx % self.batch_size == 0:
            self.end = time.time()
            self.latency = self.end - self.start
            self.throughput = self.batch_size / self.latency
            self.start = time.time()

def evaluate_pytorch(device, model_name):
    if device == 'GPU':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'CPU':
        device = torch.device('cpu')

    x_test, y_test = load_images_from_directory(
        directory=f'datasets/SweepD1/',
        image_size=(128, 128),
        color_mode="grayscale",
        class_names=['neutral', 'selection']
    )
    # Load model
    model = torch.load(f'models/pytorch/{model_name}.pt', map_location=device)
    model.eval()

    pid = os.getpid()
    process = psutil.Process(pid)

    t_avg = []
    cpu_avg = []
    score = 0

    test_batch_size = 16
    num_batches = len(x_test) // test_batch_size

    start = time.time()

    with torch.no_grad():
        for i in range(num_batches):
            x_batch = x_test[i * test_batch_size:(i + 1) * test_batch_size].to(device)
            y_batch = y_test[i * test_batch_size:(i + 1) * test_batch_size].to(device)

            cpu_start = process.cpu_percent()
            since = time.time()

            # Perform the forward pass
            prediction = model(x_batch)
            t_predict = time.time() - since

            cpu_end = process.cpu_percent()

            # Convert prediction to class label
            output = torch.argmax(prediction, dim=1).cpu().numpy()
            truth = y_batch.cpu().numpy()

            score += np.sum(output == truth)
            t_avg.append(t_predict * 1000)  # Convert time to milliseconds
            cpu_avg.append(cpu_end - cpu_start)

    stop = time.time()

    print(f'total execution time: {np.round(stop - start, 3)}s - avg batch time: {np.round(np.mean(t_avg), 3)}ms '
          f'- cpu usage: {np.round(np.mean(cpu_avg))}% - mem usage: {process.memory_percent()}% '
          f'- handles: {process.num_handles()} - threads: {process.num_threads()} - score: {score / len(x_test) * 100}%')

    return np.round(np.mean(t_avg), 3)

def evaluate_tf2(device, model_name):
    if device == 'GPU':
        pass
    elif device == 'CPU':
        tf.config.set_visible_devices([], "GPU")

    x_test, y_test = load_images_from_directory(
        directory=f'datasets/SweepD1/',
        image_size=(128, 128),
        color_mode="grayscale",
        class_names=['neutral', 'selection']
    )

    # Load model
    model = tf.keras.models.load_model(f'models/tensorflow2/{model_name}.h5', compile=False)

    pid = os.getpid()
    process = psutil.Process(pid)
    # print('start', process.num_handles(), process.num_threads())
    t_avg = []
    cpu_avg = []
    score = 0
    test_batch = batch_generator(x_test, y_test, 16)
    start = time.time()
    for x, y in test_batch:
        x = np.expand_dims(x, axis=-1)
        # print(process.num_handles(), process.num_threads())

        cpu_start = process.cpu_percent()
        since = time.time()
        prediction = model.predict(x, verbose=0)
        t_predict = time.time() - since
        cpu_end = process.cpu_percent()
        output = np.array([np.argmax(p) for p in prediction])
        truth = np.array([np.argmax(t) for t in y])

        score += np.sum(output == truth)
        t_avg.append(t_predict * 1000)
        cpu_avg.append(cpu_end - cpu_start)
    stop = time.time()
    # print('stop', process.num_handles(), process.num_threads())
    print(
        f'total execution time: {np.round(stop - start, 3)}s - avg batch time: {np.round(np.mean(t_avg), 3)}ms  - cpu usage: {np.round(np.mean(cpu_avg))} - mem usage: {process.memory_percent()} - handles: {process.num_handles()} - threads: {process.num_threads()} - score: {score / 2000 * 100}%')
    return np.round(np.mean(t_avg), 3)


def evaluate(device, model_type, model_name, repetitions=10):
    t = 0
    if model_type == 'tensorflow2':
        for i in range(repetitions):
            t += evaluate_tf2(device, model_name)
    if model_type == 'pytorch':
        for i in range(repetitions):
            t += evaluate_pytorch(device, model_name)
    print(f'average total time:  {t / repetitions}s')


if __name__ == '__main__':
    # model_type = 'tensorflow2'
    # model_name = 'model_128_avgpool_2_avgpool_2_avgpool_2_avgpool_2_D1_100'
    # device = 'CPU'

    model_type = 'pytorch'
    model_name = 'sweepnetD2'
    device = 'CPU'

    evaluate(device, model_type, model_name)

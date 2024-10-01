import tensorflow as tf
import time
import numpy as np
import psutil

from datasets.data import load_images_from_directory, batch_generator

# Disables eager execution
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


def evaluate_tf2(device, model_name):
    if device == 'GPU':
        pass
    elif device == 'CPU':
        tf.config.set_visible_devices([], "GPU")

    x_test, y_test = load_images_from_directory(
        directory=f'datasets/SweepD1/',
        image_size=(256, 256),
        color_mode="grayscale",
        class_names=['neutral', 'selection']
    )

    # Load model
    model = tf.keras.models.load_model(f'models/tensorflow2/{model_name}.h5')

    process = psutil.Process()

    t_avg = []
    cpu_avg = []
    score = 0
    test_batch = batch_generator(x_test, y_test, 16)
    start = time.time()
    for x, y in test_batch:
        x = np.expand_dims(x, axis=-1)

        cpu_start = process.cpu_percent()
        since = time.time()
        prediction = model.predict(x, verbose=1)
        t_predict = time.time() - since
        cpu_end = process.cpu_percent()
        output = np.array([np.argmax(p) for p in prediction])
        truth = np.array([np.argmax(t) for t in y])

        score += np.sum(output == truth)
        t_avg.append(t_predict * 1000)
        cpu_avg.append(cpu_end - cpu_start)
    stop = time.time()
    print(f'total execution time: {np.round(stop - start, 3)}s - avg batch time: {np.round(np.mean(t_avg), 3)}ms  - cpu usage: {np.round(np.mean(cpu_avg))} - score: {score / 2000 * 100}%')
    return np.round(np.mean(t_avg), 3)



def evaluate(device, model_type, model_name, repetitions=10):
    t = 0
    if model_type == 'tensorflow2':
        for i in range(repetitions):
            t += evaluate_tf2(device, model_name)
    print(f'average total time:  {t/repetitions}s')



if __name__ == '__main__':
    model_type = 'tensorflow2'
    model_name = 'my_model_combo22_D1'
    device = 'CPU'

    evaluate(device, model_type, model_name)
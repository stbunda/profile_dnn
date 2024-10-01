import tensorflow as tf
import time
import numpy as np

from datasets.data import load_images_from_directory, batch_generator

# Disables eager execution
tf.compat.v1.disable_eager_execution()


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

    t_avg = []
    score = 0
    test_batch = batch_generator(x_test, y_test, 16)
    start = time.time()
    for x, y in test_batch:
        x = np.expand_dims(x, axis=-1)

        since = time.time()
        prediction = model.predict(x, verbose=1)
        t_predict = time.time() - since
        output = np.array([np.argmax(p) for p in prediction])
        truth = np.array([np.argmax(t) for t in y])

        score += np.sum(output == truth)
        t_avg.append(t_predict * 1000)
    stop = time.time()
    print(f'total execution time: {np.round(stop - start, 3)}s - avg batch time: {np.round(np.mean(t_avg), 3)}ms  - score: {score / 2000 * 100}%')
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
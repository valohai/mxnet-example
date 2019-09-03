# Original: https://mxnet.incubator.apache.org/tutorials/python/mnist.html
import argparse
import gzip
import json
import logging
import os
import struct

import mxnet as mx
import numpy as np
import mxnet.runtime

fs = mx.runtime.Features()

context = mx.cpu()
if fs.is_enabled('CUDA'):
    print('CUDA enabled, using GPU context.')
    context = mx.gpu()
else:
    print('CUDA not enabled, running on CPU.')
    context = mx.cpu()

inputs_dir = os.getenv('VH_INPUTS_DIR', './inputs')
outputs_dir = os.getenv('VH_OUTPUTS_DIR', '.outputs')


def load_data():
    (train_labels, train_images) = read_data(
        os.path.join(inputs_dir, 'training-set-labels/train-labels-idx1-ubyte.gz'),
        os.path.join(inputs_dir, 'training-set-images/train-images-idx3-ubyte.gz'),
    )
    (test_labels, test_images) = read_data(
        os.path.join(inputs_dir, 'test-set-labels/t10k-labels-idx1-ubyte.gz'),
        os.path.join(inputs_dir, 'test-set-images/t10k-images-idx3-ubyte.gz'),
    )
    return {
        'train_data': train_images,
        'train_label': train_labels,
        'test_data': test_images,
        'test_label': test_labels,
    }


def read_data(labels_file, images_file):
    with gzip.open(labels_file) as file:
        struct.unpack(">II", file.read(8))
        label = np.fromstring(file.read(), dtype=np.int8)
    with gzip.open(images_file, 'rb') as file:
        _, _, rows, cols = struct.unpack(">IIII", file.read(16))
        image = np.fromstring(file.read(), dtype=np.uint8).reshape(
            len(label), rows, cols
        )
        image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32) / 255
    return label, image


def train(mnist, flags):
    batch_size = 100
    train_iter = mx.io.NDArrayIter(
        mnist['train_data'], mnist['train_label'], batch_size, shuffle=True
    )
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)

    # The first fully-connected layer and the corresponding activation function
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")

    # The second fully-connected layer and the corresponding activation function
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")

    # MNIST has 10 classes
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module
    mlp_model = mx.mod.Module(symbol=mlp, context=context)
    mlp_model.fit(
        train_iter,  # train data
        eval_data=val_iter,  # validation data
        optimizer='sgd',  # use SGD to train
        optimizer_params={'learning_rate': flags.learning_rate},
        eval_metric='acc',  # report accuracy during training
        epoch_end_callback=lambda *args: print_accuracy(mlp_model, val_iter),
        # output progress for each 100 data batches
        batch_end_callback=mx.callback.Speedometer(batch_size, 100),
        num_epoch=flags.max_epochs,
    )

    test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
    prob = mlp_model.predict(test_iter)
    assert prob.shape == (10000, 10)

    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    # predict accuracy of mlp
    acc = mx.metric.Accuracy()
    mlp_model.score(test_iter, acc)
    print_accuracy(mlp_model, test_iter, 'final_accuracy')

    save_model(mlp_model, flags.max_epochs)


def print_accuracy(model, iter, title='accuracy'):
    acc = mx.metric.Accuracy()
    model.score(iter, acc)
    print(json.dumps({title: acc.get()[1]}))


def save_model(model, epoch):
    path = os.path.join(outputs_dir, 'mnist')
    model.save_checkpoint(path, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_epochs', type=int, default=10, help='Number of epochs to train'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.1, help='Initial learning rate'
    )
    flags, _ = parser.parse_known_args()

    mnist = load_data()
    train(mnist, flags)

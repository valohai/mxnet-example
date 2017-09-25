# Original: https://mxnet.incubator.apache.org/tutorials/python/mnist.html

import json
import logging

import mxnet as mx


def print_accuracy(model, iter, title='accuracy'):
    acc = mx.metric.Accuracy()
    model.score(iter, acc)
    print(json.dumps({title: acc.get()[1]}))


def train():
    mnist = mx.test_utils.get_mnist()

    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
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
    # create a trainable module on CPU
    mlp_model = mx.mod.Module(symbol=mlp, context=mx.gpu(0))
    mlp_model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  optimizer='sgd',  # use SGD to train
                  optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
                  eval_metric='acc',  # report accuracy during training
                  epoch_end_callback=lambda *args: print_accuracy(mlp_model, val_iter),
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),  # output progress for each 100 data batches
                  num_epoch=10)  # train for at most 10 dataset passes

    test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
    prob = mlp_model.predict(test_iter)
    assert prob.shape == (10000, 10)

    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    # predict accuracy of mlp
    acc = mx.metric.Accuracy()
    mlp_model.score(test_iter, acc)
    print_accuracy(mlp_model, test_iter, 'final_accuracy')
    assert acc.get()[1] > 0.96


if __name__ == '__main__':
    train()

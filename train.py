import os
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import next_experiment_path
from batch_generator import BatchGenerator


# TODO: add help info
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', dest='seq_len', default=256, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=30, type=int)
parser.add_argument('--window_mixtures', dest='window_mixtures', default=10, type=int)
parser.add_argument('--output_mixtures', dest='output_mixtures', default=20, type=int)
parser.add_argument('--lstm_layers', dest='lstm_layers', default=3, type=int)
parser.add_argument('--units_per_layer', dest='units', default=400, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
args = parser.parse_args()

epsilon = 1e-8


class WindowLayer(object):
    def __init__(self, num_mixtures, sequence, num_letters):
        self.sequence = sequence  # one-hot encoded sequence of characters -- [batch_size, length, num_letters]
        self.seq_len = tf.shape(sequence)[1]
        self.num_mixtures = num_mixtures
        self.num_letters = num_letters
        self.u_range = -tf.expand_dims(tf.expand_dims(tf.range(0., tf.cast(self.seq_len, dtype=tf.float32)), axis=0),
                                       axis=0)

    def __call__(self, inputs, k, reuse=None):
        with tf.variable_scope('window', reuse=reuse):
            alpha = tf.layers.dense(inputs, self.num_mixtures, activation=tf.exp,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='alpha')
            beta = tf.layers.dense(inputs, self.num_mixtures, activation=tf.exp,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='beta')
            kappa = tf.layers.dense(inputs, self.num_mixtures, activation=tf.exp,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='kappa')

            a = tf.expand_dims(alpha, axis=2)
            b = tf.expand_dims(beta, axis=2)
            k = tf.expand_dims(k + kappa, axis=2)

            phi = tf.exp(-np.square(self.u_range + k) * b) * a  # [batch_size, mixtures, length]
            phi = tf.reduce_sum(phi, axis=1, keep_dims=True)  # [batch_size, 1, length]

            # whether or not network finished generating sequence
            finish = tf.cast(phi[:, 0, -1] > tf.reduce_max(phi[:, 0, :-1], axis=1), tf.float32)

            return tf.squeeze(tf.matmul(phi, self.sequence), axis=1), \
                   tf.squeeze(k, axis=2), \
                   tf.squeeze(phi, axis=1), \
                   tf.expand_dims(finish, axis=1)

    @property
    def output_size(self):
        return [self.num_letters, self.num_mixtures, 1]


class MixtureLayer(object):
    def __init__(self, input_size, output_size, num_mixtures):
        self.input_size = input_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures

    def __call__(self, inputs, bias=0., reuse=None):
        with tf.variable_scope('mixture_output', reuse=reuse):
            e = tf.layers.dense(inputs, 1,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='e')
            pi = tf.layers.dense(inputs, self.num_mixtures,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='pi')
            mu1 = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='mu1')
            mu2 = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='mu2')
            std1 = tf.layers.dense(inputs, self.num_mixtures,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='std1')
            std2 = tf.layers.dense(inputs, self.num_mixtures,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='std2')
            rho = tf.layers.dense(inputs, self.num_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075), name='rho')

            return tf.nn.sigmoid(e), \
                   tf.nn.softmax(pi * (1. + bias)), \
                   mu1, mu2, \
                   tf.exp(std1 - bias), tf.exp(std2 - bias), \
                   tf.nn.tanh(rho)


class RNNModel(tf.nn.rnn_cell.RNNCell):
    def __init__(self, layers, num_units, input_size, num_letters, batch_size, window_layer):
        super(RNNModel, self).__init__()
        self.layers = layers
        self.num_units = num_units
        self.input_size = input_size
        self.num_letters = num_letters
        self.window_layer = window_layer
        self.last_phi = None

        with tf.variable_scope('rnn', reuse=None):
            self.lstms = [tf.nn.rnn_cell.LSTMCell(num_units)
                          for _ in range(layers)]
            self.states = [tf.Variable(tf.zeros([batch_size, s]), trainable=False)
                           for s in self.state_size]

            self.zero_states = tf.group(*[sp.assign(sc)
                                          for sp, sc in zip(self.states,
                                                            self.zero_state(batch_size, dtype=tf.float32))])

    @property
    def state_size(self):
        return [self.num_units] * self.layers * 2 + self.window_layer.output_size

    @property
    def output_size(self):
        return [self.num_units]

    def call(self, inputs, state, **kwargs):
        # state[-3] --> window
        # state[-2] --> k
        # state[-1] --> finish
        # state[2n] --> h
        # state[2n+1] --> c
        window, k, finish = state[-3:]
        output_state = []
        prev_output = []

        for layer in range(self.layers):
            x = tf.concat([inputs, window] + prev_output, axis=1)
            with tf.variable_scope('lstm_{}'.format(layer)):
                output, s = self.lstms[layer](x, tf.nn.rnn_cell.LSTMStateTuple(state[2 * layer],
                                                                               state[2 * layer + 1]))
                prev_output = [output]
            output_state += [*s]

            if layer == 0:
                window, k, self.last_phi, finish = self.window_layer(output, k)

        return output, output_state + [window, k, finish]


def create_graph(num_letters, batch_size,
                 num_units=400, lstm_layers=3,
                 window_mixtures=10, output_mixtures=20):
    graph = tf.Graph()
    with graph.as_default():
        coordinates = tf.placeholder(tf.float32, shape=[None, None, 3])
        sequence = tf.placeholder(tf.float32, shape=[None, None, num_letters])
        reset = tf.placeholder(tf.float32, shape=[None, 1])
        bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

        def create_model(generate=None):
            in_coords = coordinates[:, :-1, :]
            out_coords = coordinates[:, 1:, :]

            _batch_size = 1 if generate else batch_size
            if generate:
                in_coords = coordinates

            with tf.variable_scope('model', reuse=generate):
                window = WindowLayer(num_mixtures=window_mixtures, sequence=sequence, num_letters=num_letters)

                rnn_model = RNNModel(layers=lstm_layers, num_units=num_units,
                                     input_size=3, num_letters=num_letters,
                                     window_layer=window, batch_size=_batch_size)

                reset_states = tf.group(*[state.assign(state * reset)
                                          for state in rnn_model.states])

                outs, states = tf.nn.dynamic_rnn(rnn_model, in_coords,
                                                 initial_state=rnn_model.states)

                output_layer = MixtureLayer(input_size=num_units, output_size=2,
                                            num_mixtures=output_mixtures)

                with tf.control_dependencies([sp.assign(sc) for sp, sc in zip(rnn_model.states, states)]):
                    with tf.name_scope('prediction'):
                        outs = tf.reshape(outs, [-1, num_units])
                        e, pi, mu1, mu2, std1, std2, rho = output_layer(outs, bias)

                    with tf.name_scope('loss'):
                        coords = tf.reshape(out_coords, [-1, 3])
                        xs, ys, es = tf.unstack(tf.expand_dims(coords, axis=2), axis=1)

                        mrho = 1 - tf.square(rho)
                        xms = (xs - mu1) / std1
                        yms = (ys - mu2) / std2
                        z = tf.square(xms) + tf.square(yms) - 2. * rho * xms * yms
                        n = 1. / (2. * np.pi * std1 * std2 * tf.sqrt(mrho)) * tf.exp(-z / (2. * mrho))
                        ep = es * e + (1. - es) * (1. - e)
                        rp = tf.reduce_sum(pi * n, axis=1)

                        loss = tf.reduce_mean(-tf.log(rp + epsilon) - tf.log(ep + epsilon))

                    if generate:
                        # save params for easier model loading and prediction
                        for param in [('coordinates', coordinates),
                                      ('sequence', sequence),
                                      ('bias', bias),
                                      ('e', e), ('pi', pi),
                                      ('mu1', mu1), ('mu2', mu2),
                                      ('std1', std1), ('std2', std2),
                                      ('rho', rho),
                                      ('phi', rnn_model.last_phi),
                                      ('window', rnn_model.states[-3]),
                                      ('kappa', rnn_model.states[-2]),
                                      ('finish', rnn_model.states[-1]),
                                      ('zero_states', rnn_model.zero_states)]:
                            tf.add_to_collection(*param)

                with tf.name_scope('training'):
                    steps = tf.Variable(0.)
                    learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True,
                                                               decay_steps=10000, decay_rate=0.5)

                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    grad, var = zip(*optimizer.compute_gradients(loss))
                    grad, _ = tf.clip_by_global_norm(grad, 3.)
                    train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)

                with tf.name_scope('summary'):
                    # TODO: add more summaries
                    summary = tf.summary.merge([
                        tf.summary.scalar('loss', loss)
                    ])

                return namedtuple('Model', ['coordinates', 'sequence', 'reset_states', 'reset', 'loss', 'train_step',
                                            'learning_rate', 'summary'])(
                           coordinates, sequence, reset_states, reset, loss, train_step, learning_rate, summary
                       )
        train_model = create_model(generate=None)
        _ = create_model(generate=True)  # just to create ops for generation

    return graph, train_model


def main():
    restore_model = args.restore
    seq_len = args.seq_len
    batch_size = args.batch_size
    num_epoch = args.epochs
    batches_per_epoch = 1000

    batch_generator = BatchGenerator(batch_size, seq_len)
    g, vs = create_graph(batch_generator.num_letters, batch_size,
                         num_units=args.units, lstm_layers=args.lstm_layers,
                         window_mixtures=args.window_mixtures,
                         output_mixtures=args.output_mixtures)

    with tf.Session(graph=g) as sess:
        model_saver = tf.train.Saver(max_to_keep=2)
        if restore_model:
            model_file = tf.train.latest_checkpoint(os.path.join(restore_model, 'models'))
            experiment_path = restore_model
            epoch = int(model_file.split('-')[-1]) + 1
            model_saver.restore(sess, model_file)
        else:
            sess.run(tf.global_variables_initializer())
            experiment_path = next_experiment_path()
            epoch = 0

        summary_writer = tf.summary.FileWriter(experiment_path, graph=g, flush_secs=10)
        summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),
                                       global_step=epoch * batches_per_epoch)

        for e in range(epoch, num_epoch):
            print('\nEpoch {}'.format(e))
            for b in range(1, batches_per_epoch + 1):
                coords, seq, reset, needed = batch_generator.next_batch()
                if needed:
                    sess.run(vs.reset_states, feed_dict={vs.reset: reset})
                l, s, _ = sess.run([vs.loss, vs.summary, vs.train_step],
                                   feed_dict={vs.coordinates: coords,
                                              vs.sequence: seq})
                summary_writer.add_summary(s, global_step=e * batches_per_epoch + b)
                print('\r[{:5d}/{:5d}] loss = {}'.format(b, batches_per_epoch, l), end='')

            model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'),
                             global_step=e)


if __name__ == '__main__':
    main()

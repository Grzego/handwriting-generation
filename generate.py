import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation
import seaborn
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--style', dest='style', type=int, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--animation', dest='animation', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
parser.add_argument('--save', dest='save', type=str, default=None)
args = parser.parse_args()


def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, style=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: args.bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not args.force and finish[0, 0] > 0.8:
                print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        while True:
            if args.text is not None:
                args_text = args.text
            else:
                args_text = input('What to generate: ')

            style = None
            if args.style is not None:
                style = None
                with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
                    styles = pickle.load(file)

                if args.style > len(styles[0]):
                    raise ValueError('Requested style is not in style list')

                style = [styles[0][args.style], styles[1][args.style]]

            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, args_text, translation, style)

            strokes = np.array(stroke_data)
            epsilon = 1e-8
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
            minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
            miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

            if args.info:
                delta = abs(maxx - minx) / 400.
                x = np.arange(minx, maxx, delta)
                y = np.arange(miny, maxy, delta)
                x_grid, y_grid = np.meshgrid(x, y)
                z_grid = np.zeros_like(x_grid)
                for i in range(strokes.shape[0]):
                    gauss = mlab.bivariate_normal(x_grid, y_grid, mux=strokes[i, 0], muy=strokes[i, 1],
                                                  sigmax=strokes[i, 2], sigmay=strokes[i, 3],
                                                  sigmaxy=0.)  # strokes[i, 4]
                    z_grid += gauss * np.power(strokes[i, 2] + strokes[i, 3], 0.4) / (np.max(gauss) + epsilon)

                fig, ax = plt.subplots(2, 2)

                ax[0, 0].imshow(z_grid, interpolation='bilinear', aspect='auto', cmap=cm.jet)
                ax[0, 0].grid(False)
                ax[0, 0].set_title('Densities')
                ax[0, 0].set_aspect('equal')

                for stroke in split_strokes(cumsum(np.array(coords))):
                    ax[0, 1].plot(stroke[:, 0], -stroke[:, 1])
                ax[0, 1].set_title('Handwriting')
                ax[0, 1].set_aspect('equal')

                phi_img = np.vstack(phi_data).T[::-1, :]
                ax[1, 0].imshow(phi_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
                ax[1, 0].set_yticks(np.arange(0, len(args_text) + 1))
                ax[1, 0].set_yticklabels(list(' ' + args_text[::-1]), rotation='vertical', fontsize=8)
                ax[1, 0].grid(False)
                ax[1, 0].set_title('Phi')

                window_img = np.vstack(window_data).T
                ax[1, 1].imshow(window_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
                ax[1, 1].set_yticks(np.arange(0, len(charset)))
                ax[1, 1].set_yticklabels(list(charset), rotation='vertical', fontsize=8)
                ax[1, 1].grid(False)
                ax[1, 1].set_title('Window')

                plt.show()
            else:
                fig, ax = plt.subplots(1, 1)
                for stroke in split_strokes(cumsum(np.array(coords))):
                    plt.plot(stroke[:, 0], -stroke[:, 1])
                ax.set_title('Handwriting')
                ax.set_aspect('equal')
                plt.show()

            if args.animation:
                fig, ax = plt.subplots(1, 1, frameon=False, figsize=(2 * (maxx - minx + 2) / (maxy - miny + 1), 2))
                ax.set_xlim(minx - 1., maxx + 1.)
                ax.set_ylim(-maxy - 0.5, -miny + 0.5)
                ax.set_aspect('equal')
                ax.axis('off')
                # ax.hold(True)

                plt.draw()
                plt.show(False)

                background = fig.canvas.copy_from_bbox(ax.bbox)

                sumed = cumsum(coords)

                def _update(i):
                    c1, c2 = sumed[i: i+2]
                    fig.canvas.restore_region(background)
                    if c1[2] == 1. and c2[2] == 1.:
                        line, = ax.plot([c2[0], c2[0]], [-c2[1], -c2[1]])
                    elif c1[2] != 1.:
                        line, = ax.plot([c1[0], c2[0]], [-c1[1], -c2[1]])
                    else:
                        line, = ax.plot([c1[0], c1[0]], [-c1[1], -c1[1]])
                    fig.canvas.blit(ax.bbox)
                    return line,

                anim = animation.FuncAnimation(fig, _update, frames=len(sumed) - 2,
                                               interval=16, blit=True, repeat=False)
                if args.save is not None:
                    anim.save(args.save, fps=60, extra_args=['-vcodec', 'libx264'])
                plt.show()

            if args.text is not None:
                break


if __name__ == '__main__':
    main()

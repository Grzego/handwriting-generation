import os
import html
import pickle
import numpy as np
import xml.etree.cElementTree as ElementTree

"""
- all points:
>> [[x1, y1, e1], ..., [xn, yn, en]]
- indexed values
>> [h1, ... hn]
"""


def distance(p1, p2, axis=None):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def clear_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = distance(p1, p2) + distance(p2, p3)
        if dist > 1500:
            to_remove.add(i)
    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600:
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]


def main():
    data = []
    charset = set()

    file_no = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == '.xml':
                file_no += 1
                print('[{:5d}] File {} -- '.format(file_no, os.path.join(root, file)), end='')
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')
                if not transcription:
                    print('skipped')
                    continue
                texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
                strokes = []
                mid_points = []
                for ps in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in ps])
                    pts[-1, 2] = 1

                    pts = clear_middle(pts)
                    if len(pts) == 0:
                        continue

                    seps = separate(pts)
                    for pss in seps:
                        if len(seps) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[0], min(pss, key=lambda x: x[1])[1]

                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]
                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]
                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)
                lines = []
                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]
                print('lines = {:4d}; texts = {:4d}'.format(len(lines), len(texts)))
                charset |= set(''.join(texts))
                data += [(texts, lines)]
    print('data = {}; charset = ({}) {}'.format(len(data), len(charset), ''.join(sorted(charset))))

    translation = {'<NULL>': 0}
    for c in ''.join(sorted(charset)):
        translation[c] = len(translation)

    def translate(txt):
        return list(map(lambda x: translation[x], txt))

    dataset = []
    labels = []
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]
            labels += [translate(text)]

    whole_data = np.concatenate(dataset, axis=0)
    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

    try:
        os.makedirs('data')
    except FileExistsError:
        pass
    np.save(os.path.join('data', 'dataset'), np.array(dataset))
    np.save(os.path.join('data', 'labels'), np.array(labels))
    with open(os.path.join('data', 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)


if __name__ == '__main__':
    main()

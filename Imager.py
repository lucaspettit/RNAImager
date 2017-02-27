import pygame
from PIL import Image
import numpy
from AI import perceptron

import queue


class Node(object):
    x = None
    y = None
    value = None
    neighbors = []

    def __init__(self, value, coord):
        self.value = value
        self.x, self.y = coord

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value


class RemoveToFit(object):
    # RemoveToFit - removes the lowest and highest values from the dataset and
    # replace them with the average value until the std and difference between
    # the mean & median is below a threshold.
    # This algorithm will store all the upper bounded pixels and do a post-evaluation.
    #
    # Post-evaluation - Data is stored in a forest of grids where it will determine the
    # average well size and search the grids for wells that fit that description.

    _original_image = None
    _scaler = 1
    __regions = ([], [])
    _trim_count = 0
    __data = []
    _ai = None

    def __init__(self, debug=False):
        self.__debug = debug
        self._ai = perceptron()

    def image(self, viewRaw=False, fullSpectrum=False):
        return self._render(viewRawData=viewRaw, viewFullSpectrum=fullSpectrum)

    def reset(self):
        self.__regions = ([], [])
        self.__data = []
        self._trim_count = 0

    def eval(self, img, COMPW=128):
        self._original_image = img
        seed, cmpImage = self._prepData(img, COMPW)
        self.__data = seed

        forest = self._plantForest(seed)

        regions = self._buildRegions(forest)
        acc, rej = self._categorize(regions, cmpImage)
        self.__regions = acc, rej

        print('Done!')

    def _prepData(self, image, COMPW=128, debug=False):
        if debug:
            print('Compressing...')

        # compress image
        w, h = image.size
        if COMPW > 0:
            cw = (COMPW / float(w))
            ch = int((float(h) * float(cw)))
            cw = COMPW
            image = image.resize((cw, ch), Image.ANTIALIAS)
            cw, ch = image.size
            self._scaler = w/COMPW
        else:
            cw, ch = w, h

        # transform data
        if debug:
            print('new image dimensions: ' + str(cw) + ', ' + str(ch))
            print('transforming data...')

        pixels = []
        pdata = list(image.getdata(band=1))
        cmpImage = numpy.zeros([cw, ch])
        for i in range(int(ch * cw)):
            pixel = pdata[i]
            py = int(i % cw)
            px = int(i / cw)
            pixels.append(Node(pixel, (px, py)))
            cmpImage[py, px] = pixel

        cmpImage = numpy.uint8(cmpImage)
        raw = numpy.sort(numpy.array(pixels))
        f = []

        count, raw, f = self._trim(0, raw, f)
        self._trim_count = count

        grid = numpy.zeros([ch, cw])
        x, y = 0, 0
        for n in f:
            x = max(n.x, x)
            y = max(n.y, y)
            grid[n.x, n.y] = n.value

        if debug:
            print('Done!')

        return grid, cmpImage

    def _trim(self, iteration, pool, bucket):

        iteration += 1
        length = len(pool)
        if length == 0:
            return iteration, pool, bucket
        mean, median, minimum, maximum = self._stats(pool)

        toTrim_L = minimum + int(0.1 * (mean - minimum))
        toTrim_U = maximum - int(0.1 * (maximum - mean))

        lower_count = 0
        upper_count = 0

        for lower_count in range(length):
            node = pool[lower_count]
            if node.value > toTrim_L:
                break
            bucket.append(Node(-int(iteration), (node.x, node.y)))

        for upper_count in range(length):
            node = pool[-upper_count-1]
            if node.value < toTrim_U:
                break
            bucket.append(Node(int(iteration), (node.x, node.y)))

        pool = pool[lower_count:-upper_count-1]

        return self._trim(iteration + 1, pool, bucket)

    def _plantForest(self, seed):
        forest = []
        w, h = seed.shape
        checked = {}

        for x in range(w):
            for y in range(h):
                value = seed[(x, y)]
                if value > 0 and value not in checked:
                    q = queue.Queue()
                    q.put((x, y))
                    tree = []
                    checked[(x, y)] = True

                    while not q.empty():
                        i, j = q.get()
                        tree.append([i, j])

                        for di, dj in self._neighbors((i, j), (w, h)):
                            if (di, dj) not in checked and seed[di, dj] > 0:
                                checked[(di, dj)] = True
                                q.put((di, dj))

                    if len(tree) > 4:
                        forest.append(numpy.array(tree))

        return forest

    def _buildRegions(self, forest):

        regions = []
        max_x, max_y = self.__data.shape

        for f in forest:
            x = f[:, 0]
            y = f[:, 1]

            w = numpy.max(x) - numpy.min(x)
            h = numpy.max(y) - numpy.min(y)
            x = numpy.min(x) - (((w * 2) - w) / 2)
            y = numpy.min(y) - (((h * 2) - h) / 2)
            w *= 2
            h *= 2

            # make sure everything is inside of the image
            x, y, w, h = int(x), int(y), int(w), int(h)

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > max_x:
                w = int(max_x - x)
            if y + h > max_y:
                h = int(max_y - y)

            rect = ((y, x), (h, w))

            regions.append(rect)

        return regions

    def _categorize(self, regions, grid):
        if len(regions) == 0:
            return []
        acc = []
        rej = []
        ave = []
        normalized = self._normalize_region(regions)

        for n, r in zip(normalized, regions):
            coord, dem = n
            x, y = coord
            w, h = dem

            snippet = self._original_image.crop((x, y, x+w, y+h))
            w, h = snippet.size
            scailer = 28 / max(w, h)
            squariness = (abs(w-h)/max(w, h))
            features = [int(w * scailer), int(h * scailer), float(squariness)]
            features += snippet.resize((28, 28)).getdata(band=1)

            if self._ai.predict(features) > 0:
                ave.append([w, h])
                acc.append(r)
            else:
                rej.append(r)

        return acc, rej

    def _neighbors(self, coord, size):
        x, y = coord
        w, h = size
        n = []
        if x - 1 >= 0:
            n.append((x - 1, y))
        if x + 1 < w:
            n.append((x + 1, y))
        if y - 1 >= 0:
            n.append((x, y - 1))
        if y + 1 < h:
            n.append((x, y + 1))
        if x - 1 >= 0 and y - 1 >= 0:
            n.append((x - 1, y - 1))
        if x - 1 >= 0 and y + 1 < h:
            n.append((x - 1, y + 1))
        if x + 1 < w and y - 1 >= 0:
            n.append((x + 1, y - 1))
        if x + 1 < w and y + 1 < h:
            n.append((x + 1, y + 1))

        return n

    def _stats(self, data):
        mean = float(0)
        smallest, largest = 255, 0

        for node in data:
            value = node.value
            mean += value
            smallest = min(smallest, value)
            largest = max(largest, value)

        mean /= float(len(data))
        median = ((largest - smallest) / 2) + smallest
        return mean, median, smallest, largest

    def _normalize_region(self, region):
        _r = []
        _w, _h = self._original_image.size
        for coord, dem in region:
            x, y = coord
            w, h = dem
            x, y = int(x * self._scaler), int(y * self._scaler)
            w, h = int(w * self._scaler), int(h * self._scaler)
            if x < 0: x = 0
            if y < 0: y = 0
            if w >= _w: w = _w - 1
            if h >= _h: h = _h - 1
            _r.append(((x, y), (w, h)))
        return _r

    def _render(self, viewRawData=False, viewFullSpectrum=False):

        if viewRawData:
            rb = int(255 / (1.5 * self._trim_count))
            g = int(255 / self._trim_count)

            w, h = self.__data.shape
            grid = numpy.zeros((w, h, 3))

            for x in range(w):
                for y in range(h):
                    v = self.__data[x, y]
                    if v > 0:
                        color = max((255 - (v * rb)), 0), min((v * g), 255), 0
                        grid[x, y] = color
                    elif viewFullSpectrum and v < 0:
                        v = -v
                        grid[x, y] = 0, min((v * g), 255), max(255 - (v * rb), 0)
            acc, rej = self.__regions
            return Image.fromarray(numpy.uint8(grid)), acc, rej

        _a, _r = self.__regions
        acc = self._normalize_region(_a)
        rej = self._normalize_region(_r)

        return self._original_image, acc, rej

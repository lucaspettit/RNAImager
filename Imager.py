import pygame
from PIL import Image
import numpy

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

    __data_prepped = False
    __iteration = 0

    __forest = []
    __regions = []

    __COMPW = 256
    __comp_w, __comp_h = 0, 0
    _trim_count = 0

    __debug = False
    __viewFullSpectrum = True

    __compressed_image = None
    __image = None
    __data = []
    __data_len = 0

    __original_size = 0, 0

    __min_cluster_size = 0

    def __init__(self, debug=False):
        self.__debug = debug

    def image(self):
        return self._render(self.__data)

    def reset(self):
        self.__forest = []
        self.__regions = []
        self.__compressed_image = None
        self.__comp_h = 0
        self.__data = []
        self.__data_prepped = False
        self.__iteration = 0
        self._trim_count = 0

    def compressionRatio(self, cr=None):
        if cr is None:
            return self.__comp_w
        elif cr >= 0:
            self.__COMPW = cr
        else:
            print('Invalid compression ratio: ' + str(cr))
            raise SystemExit

    def step(self):
        if self.__data_prepped:
            self.__trim__(self.__iteration)
            self.__iteration += 1

    def eval(self, img, COMPW=128):
        seed = self._prepData(img, COMPW)
        self.__data = seed

        forest = self._plantForest(seed)

        regions = self._buildRegions(forest)
        self.__regions = regions

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
        else:
            cw, ch = w, h

        # transform data
        if debug:
            print('new image dimensions: ' + str(cw) + ', ' + str(ch))
            print('transforming data...')

        pixels = []
        pdata = list(image.getdata(band=1))
        for i in range(int(ch * cw)):
            pixel = pdata[i]
            py = int(i % cw)
            px = int(i / cw)
            pixels.append(Node(pixel, (px, py)))

        raw = numpy.sort(numpy.array(pixels))
        f = []

        count, raw, f = self._trim(0, raw, f)
        self._trim_count = count

        grid = numpy.zeros((ch, cw))
        x, y = 0, 0
        for n in f:
            x = max(n.x, x)
            y = max(n.y, y)
            grid[n.x, n.y] = n.value

        if debug:
            print('Done!')

        return grid

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

        for f in forest:
            x = f[:, 0]
            y = f[:, 1]
            mx = numpy.mean(x)
            stdx = numpy.std(x)
            x = mx - (2 * stdx)
            w = 4 * stdx

            my = numpy.mean(y)
            stdy = numpy.std(y)
            y = my - (2 * stdy)
            h = 4 * stdy

            rect = ((y, x), (h, w))

            regions.append(rect)

        return regions

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

    def _render(self, data, viewFullSpectrum=False):

        rb = int(255 / (1.5 * self._trim_count))
        g = int(255 / self._trim_count)

        w, h = data.shape
        grid = numpy.zeros((w, h, 3))

        for x in range(w):
            for y in range(h):
                v = data[x, y]
                if v > 0:
                    color = max((255 - (v * rb)), 0), min((v * g), 255), 0
                    grid[x, y] = color
                elif viewFullSpectrum and v < 0:
                    v = -v
                    grid[x, y] = 0, min((v * g), 255), max(255 - (v * rb), 0)

        return Image.fromarray(numpy.uint8(grid)), self.__regions

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

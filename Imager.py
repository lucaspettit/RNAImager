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

    __COMPW = 256
    __comp_w, __comp_h = 0, 0
    __total_trim_iterations = 0

    __debug = False
    __viewFullSpectrum = True

    __compressed_image = None
    __image = None
    __data = []
    __data_len = 0

    __min_cluster_size = 0

    def __init__(self, debug=False):
        self.__debug = debug

    def image(self, path=None):
        if path is None:
            return self.__render__(self.__viewFullSpectrum)
        else:
            try:
                self.reset()
                self.__image = Image.open(path)
                return True
            except pygame.error:
                print('Unable to open image: ' + path)
                return False

    def reset(self):
        self.__forest = []
        self.__compressed_image = None
        self.__comp_h = 0
        self.__data = []
        self.__data_prepped = False
        self.__iteration = 0
        self.__total_trim_iterations = 0

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

    def eval(self, img=None):
        if img != None:
            if not self.image(img):
                return False

        if self.__image is None:
            print('No Image to evaluate')
            return False

        if not self.__transform_data__():
            return False

        i=0
        while self.__trim__(i):
            i += 1

        #self.__remove_noise__()

        #if self.__debug:
        #    print('Completed calculation')
        #    print(str(self.__total_trim_iterations) + ' total iterations')

    def __transform_data__(self):
        # Transforms self.__image into a numpy array of Nodes
        if self.__debug:
            print('Compressing...')

        # compress image
        w, h = self.__image.size
        if self.__COMPW > 0:
            cw = (self.__COMPW / float(w))
            ch = int((float(h) * float(cw)))
            cw = self.__COMPW
            self.__image = self.__image.resize((cw, ch), Image.BICUBIC)
            #ch, cw = img.size
            self.__comp_w, self.__comp_h = ch, cw
        else:
            ch, cw = self.__image.size
            self.__comp_h, self.__comp_w = self.__image.size
            img = self.__image

        # transform data
        if self.__debug:
            print('new image dimensions: ' + str(cw) + ', ' + str(ch))
            print('transforming data...')

        pixels = []
        pdata = list(self.__image.getdata(band=1))
        for i in range(int(ch * cw)):
            pixel = pdata[i]
            py = int(i % cw)
            px = int(i / cw)
            pixels.append(Node(pixel, (px, py)))

        self.__compressed_image = numpy.array(pixels)
        self.__data = numpy.sort(self.__compressed_image)
        self.__forest = []
        self.__data_prepped = True

        if self.__debug:
            print('Done!')
        return True

    def __trim__(self, iteration):
        #if self.__debug:
        #    print('Trimming...')
        #    print('Iteration: ' + str(iteration))
        #    print('Data Size: ' + str(len(self.__data)))

        iteration += 1
        length = len(self.__data)
        if length == 0:
            return False
        mean, median, minimum, maximum = self.__stats__(self.__data)

        #if self.__debug:
        #    print('mean: ' + str(mean))
        #    print('median: ' + str(median))

        toTrim_L = minimum + int(0.1 * (mean - minimum))
        toTrim_U = maximum - int(0.1 * (maximum - mean))

        #if self.__debug:
        #    print('Upper Threshold: ' + str(toTrim_U))
        #    print('Lower Threshold: ' + str(toTrim_L))

        lower_count = 0
        upper_count = 0

        #print('data range: [' + str(self.__data[0].value) + ', ' + str(self.__data[-1].value) + ']')

        for lower_count in range(length):
            node = self.__data[lower_count]
            if node.value > toTrim_L:
                break
            self.__forest.append(Node(-int(iteration), (node.x, node.y)))

        for upper_count in range(length):
            node = self.__data[-upper_count-1]
            if node.value < toTrim_U:
                break
            self.__forest.append(Node(int(iteration), (node.x, node.y)))

        self.__data = self.__data[lower_count:-upper_count-1]

        #if self.__debug:
        #    print('Removed ' + str(upper_count) + ' upper bound pixels')
        #    print('Removed ' + str(lower_count) + ' lower bound pixels')
        #    print('Done!')

        self.__total_trim_iterations += 1
        return True

    def __remove_noise__(self):
        checked = numpy.zeros(self.__data_size)

        w, h = self.__data_size
        for x in range(w):
            for y in range(h):
                if checked[x, y] != 0:
                    continue
                if self.__forest[x, y] > 0:
                    blob, checked = self.__bff__(self.__forest, checked, self.__data_size, (x, y), 0)
                    if len(blob) < self.__min_cluster_size:
                        for x, y in blob:
                            self.__forest[x, y] = 0

    def __neighbors__(self, coord, size):
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

    def __bff__(self, map, checked, size, start, thold):
        q = queue.Queue()
        q.put(start)
        blob = []
        x, y = start
        checked[x, y] = 1

        while not q.empty():
            x, y = q.get()
            blob.append((x, y))

            for dx, dy in self.__neighbors__((x, y), size):
                if checked[dx, dy] == 0 and map[dx, dy] >= thold:
                    checked[dx, dy] = 1
                    q.put((dx, dy))

        return (blob, checked)

    def __render__(self, viewFullSpectrum=False):
        if self.__compressed_image is None:
            return self.__image

        if len(self.__forest) == 0:
            print('forest len == 0')
            img = numpy.zeros((self.__comp_w, self.__comp_h))
            for node in self.__compressed_image:
                x, y = node.x, node.y
                img[x, y] = node.value
            return img

        else:
            rb = int(255 / (1.5 * self.__total_trim_iterations))
            g = int(255 / self.__total_trim_iterations)
            length = len(self.__forest)
            grid = numpy.zeros((self.__comp_w, self.__comp_h, 3))
            print('grid shape: ' + str(grid.shape))

            for node in self.__forest:
                v = node.value
                if v > 0:
                    color = max((255 - (v * rb)), 0), min((v * g), 255), 0
                    grid[node.x, node.y] = color
                elif viewFullSpectrum and v < 0:
                    v = -v
                    grid[node.x, node.y] = 0, min((v * g), 255), max(255 - (v * rb), 0)

            return Image.fromarray(numpy.uint8(grid))

    def __render__heatmap_i__(self, value):

        range = 2 * self.__total_trim_iterations
        gth_L = -int(self.__total_trim_iterations / 2)
        gth_M = 0
        gth_U = -gth_L
        rth = gth_L
        bth = gth_U

        if isinstance(value, list) or isinstance(value, set) or isinstance(value, numpy.ndarray):
            num = 0
            for v in value:
                num += v
            value = num / len(value)

        r, g, b = (0,0,0)

        if value > rth:
            r = (abs((value - rth) * 2) + 1 / range) * 255
        if value > gth_L and value < gth_M:
            g = (abs((value - gth_L) * 4) / range) * 255
        if value > gth_M and value < gth_U:
            g =  255 - (((abs(value - gth_L) * 4) / range) * 255)
        if value < bth:
            b = (abs(value * 2) / range) * 255

        return (r, g, b)

    def __stats__(self, data):
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

import pygame
import numpy

from os import listdir
from os.path import isfile, join, basename
import sys
import queue


class Node(object):
    x = None
    y = None
    value = None
    neighbors = []

    def __init__(self, value, coord):
        self.value = value
        self.x, self.y = coord

    def __get__(self, obj, objtype):
        return self.value

    def __set__(self, obj, value):
        if isinstance(value, int) or isinstance(value, float):
            self.value = value

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

    __forest = None
    __compressed_image = None

    __compression_ratio = 0
    __mm_thold = 0.15
    __std_thold = 0.5
    __total_trim_iterations = 0

    __debug = False
    __viewFullSpectrum = True

    __orig_image = None
    __data = None
    __orig_size = None
    __data_size = (0, 0)

    __min_cluster_size = 0

    def __init__(self, debug=False):
        self.__debug = debug

    def image(self, path=None):
        if path is None:
            return self.__render__(self.__viewFullSpectrum)
        else:
            try:
                self.reset()
                self.__orig_image = pygame.image.load(path)
                self.__orig_size = self.__orig_image.get_rect().size
                return True
            except pygame.error:
                print('Unable to open image: ' + path)
                return False

    def reset(self):
        self.__forest = None
        self.__compressed_image = None
        self.__data = None
        self.__data_size = (0, 0)
        self.__data_prepped = False
        self.__iteration = 0
        self.__total_trim_iterations = 0

    def compressionRatio(self, cr=None):
        if cr is None:
          return self.__compression_ratio
        elif cr >= 0:
          self.__compression_ratio = cr
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

        if self.__orig_image is None:
            print('No Image to evaluate')
            return False

        if not self.__transform_data__():
            return False

        i=0
        while(self.__trim__(i)):
            i += 1

        self.__remove_noise__()

        if self.__debug:
            print('Completed calculation')
            print(str(self.__total_trim_iterations) + ' total iterations')

    def __transform_data__(self):
        if self.__debug:
            print('Compressing...')

        w, h = self.__orig_size
        if self.__compression_ratio > 0:
            jw, jh = round(w / self.__compression_ratio), round(h / self.__compression_ratio)
            if self.__debug:
                print('original image dimensions: ' + str(w) + ', ' + str(h))
                print('compression rate: ' + str(self.__compression_ratio))
                print('jump count: ' + str(jw) + ', ' + str(jh))

            if jw == 0 or jh == 0:
                print('Exiting compression: compression ratio is grater than image dimensions')
                return False

            pixels = pygame.surfarray.array3d(self.__orig_image)
            pixels = pixels[::jw, ::jh, :]

            w, h = int(w / jw), int(h / jh)
        else:
            pixels = pygame.surfarray.array3d(self.__orig_image)

        self.__data = numpy.zeros((w, h))
        self.__forest = numpy.zeros((w, h))
        self.__compressed_image = pixels

        if self.__debug:
            print('new image dimensions: ' + str(w) + ', ' + str(h))

        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                r, g, b = float(r), float(g), float(b)
                self.__data[x, y] = int((r + g + b) / 3)

        self.__data_size = (w, h)
        self.__min_cluster_size = 18
        self._image = pygame.surfarray.make_surface(pixels)
        self.__data_prepped = True

        if self.__debug:
            print('Done!')
        return True

    def __trim__(self, iteration):
        if self.__debug:
            print('Trimming...')
            print('Iteration: ' + str(iteration))

        iteration += 1
        mean = numpy.mean(self.__data)
        median = numpy.median(self.__data)
        std = numpy.std(self.__data)

        if self.__debug:
            print('mean: ' + str(mean))
            print('median: ' + str(median))
            print('std: ' + str(std))


        if abs(mean - median) <= self.__mm_thold and std < self.__std_thold:
            if self.__debug:
                print('End threshold passed')
            return False

        minimum = numpy.min(self.__data)
        maximum = numpy.max(self.__data)

        toTrim_L = minimum + int(0.1 * (mean - minimum))
        toTrim_U = maximum - int(0.18 * (maximum - mean))

        if self.__debug:
            print('Upper Threshold: ' + str(toTrim_U))
            print('Lower Threshold: ' + str(toTrim_L))

        w, h = self.__data_size
        lower_count = 0
        upper_count = 0

        for y in range(h):
            for x in range(w):
                if self.__forest[x, y] != 0:
                    continue

                value = self.__data[x, y]
                if value <= toTrim_L:
                    self.__data[x, y] = mean
                    self.__forest[x, y] = -int(iteration)
                    lower_count += 1
                elif value >= toTrim_U:
                    self.__data[x, y] = mean
                    self.__forest[x, y] = int(iteration)
                    upper_count += 1

        if lower_count == 0 and upper_count == 0:
            if self.__debug:
                print('no pixels removed')
                print('End threshold passed')
            return False

        if self.__debug:
            print('Removed ' + str(upper_count) + ' upper bound pixels')
            print('Removed ' + str(lower_count) + ' lower bound pixels')
            print('Done!')

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
            return self.__orig_image

        a = int(255 / self.__total_trim_iterations)
        aa = int(200 / self.__total_trim_iterations)
        w, h = self.__data_size
        grid = numpy.zeros((w, h, 3))

        for x in range(w):
            for y in range(h):
                if self.__forest[x, y] > 0:
                    f = self.__forest[x, y]
                    grid[x, y] = max((255 - (f*a)), 0), min((f*aa), 200), 0

                elif viewFullSpectrum and self.__forest[x, y] < 0:
                    f = -int(self.__forest[x, y])
                    grid[x, y] = 0, min((f * a), 255), max(200 - (f*aa), 0)

                else:
                    grid[x, y] = self.__compressed_image[x, y]
                    #grid[x, y] = (0, 200, 0)

        return pygame.surfarray.make_surface(grid)

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


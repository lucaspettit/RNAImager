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
        self._ai = BestFit()

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

            w = (numpy.max(x) - numpy.min(x)) * 2
            h = (numpy.max(y) - numpy.min(y)) * 2
            x = numpy.min(x) - (0.125 * w)
            y = numpy.min(y) - (0.125 * h)

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
            # w, h = snippet.size
            # scailer = 28 / max(w, h)
            # squariness = (abs(w-h)/max(w, h))
            # features = [int(w * scailer), int(h * scailer), float(squariness)]
            # features += snippet.resize((28, 28)).getdata(band=1)

            if self._ai.predict(snippet) > 100:
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


class BestFit(object):
    # _w = numpy.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0])
    # _w = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255])
    _w = numpy.array([-4.75, -4.75, -5.75, -4.75, -4.75, 2, 2, 2, -4.75, 2, 2, 2, -4.75, 2, 2, 2])
    _base = 4
    _ws = _base, _base
    _wd = _base * _base
    _step = 2

    def eval(self, img):
        # Quarters the image and evaluates each quadrant.
        # returns a list of 5 rectangles that correspond to
        # the TL, TR, BL, BR, FULL

        w, h = img.size
        horizontal = (w > h)
        dem = min(w, h)
        NML = abs(w - h)
        cache = []

        for i in range(int(NML / self._step) + 1):
            if horizontal:
                offset_x, offset_y = i * self._step, 0
            else:
                offset_x, offset_y = 0, i * self._step

            rect = offset_x, offset_y, dem, dem
            corners = self.evalSubSection(img.crop(rect))
            bgValue = (corners['TL'][1] + corners['TR'][1] + corners['BL'][1] + corners['BR'][1]) / 4
            x, y = corners['TL'][0][0], corners['TL'][0][1]
            w, h = dem + rect[0] + corners['BR'][0][2] - x, dem + rect[1] + corners['BR'][0][3] - y
            bgRect = x, y, w, h
            corners['FULL'] = bgRect, bgValue
            cache.append((corners, (offset_x, offset_y)))

        best = dict()
        bestValue = -999999
        for corner, offset in cache:
            if bestValue < corner['FULL'][1]:
                bestVaue = corner['FULL'][1]
                best = {}
                for q in corner.keys():
                    if q == 'FULL':
                        continue
                    r, v = corner[q]
                    _x, _y = offset
                    x, y, w, h = r
                    best[q] = (x + _x, y + _y, w, h), v

        if len(best) == 0:
            print("set of corners is empty: cache size -> {0}".format(len(cache)))
        else:
            tl, tlv = best['TL']
            tr, trv = best['TR']
            bl, blv = best['BL']
            br, brv = best['BR']

            x, y = min(tl[0], bl[0]), min(tl[1], tr[1])
            w, h = max(tr[0] + tr[2], br[0] + br[2]) - x, max(br[1] + br[3], bl[1] + bl[3]) - y
            best['FULL'] = (x, y, w, h), float((tlv + trv + blv + brv)/4)
        return best

    def predict(self, img):
        sections = self.eval(img)
        if sections is None:
            print("this program totally sucks")
            return -999999
        tot = 0.0
        for s, v in sections.values():
            tot += v
        if tot == 0.0:
            return -999999
        return float(tot/len(sections))

    def evalSubSection(self, img):
        cache = dict()
        dem = img.size[0]
        quadrants = self.quarter(img)
        for id in quadrants.keys():
            featureDict = self.buildFeatureDict(self.shred(quadrants[id], self._step))
            cache[id] = featureDict
        return self.getMostSimilar(cache, int(dem/2))

    def quarter(self, img):
        w, h = img.size
        sw, sh = int(w/2), int(h/2)
        q = dict()
        tl = (0, 0, sw, sh)
        tr = (sw + 1, 0, sw + w - sw, sh)
        br = (sw + 1, sh + 1, sw + w - sw - 1, sh + h - sh - 1)
        bl = (0, sh + 1, w - sw - 1, sh + h - sh - 1)
        q['TL'] = img.crop(tl)
        q['TR'] = img.crop(tr).rotate(90)
        q['BR'] = img.crop(br).rotate(180)
        q['BL'] = img.crop(bl).rotate(270)
        return q

    def shred(self, img, skip):
        # returns a list of (rect, vector) pairs
        base = self._base
        w, h = img.size
        features = []

        for i in range(int((w - base) / skip)):
            step = i * skip + base
            dem = w - step
            rect = step, step, dem, dem
            vector = numpy.array(list(img.crop(rect).resize(self._ws, Image.ANTIALIAS).getdata(band=1)))
            features.append((rect, vector))

            rect = 0, 0, dem, dem
            vector = numpy.array(list(img.crop(rect).resize(self._ws, Image.ANTIALIAS).getdata(band=1)))
            features.append((rect, vector))

        return features

    def buildFeatureDict(self, features):
        fdict = {}
        for rect, x in features:
            fdict[rect] = numpy.dot(self._w, x)
        return fdict

    def getMostSimilar(self, quadrants, dem):
        best = dict()
        for quad in quadrants:
            value = -999999
            rect = (0, 0, 0, 0)
            for r in quadrants[quad].keys():
                if value < quadrants[quad][r]:
                    value = quadrants[quad][r]
                    rect = r

            # trying to reset the rotation... not sure if it's right
            x, y, w, h = rect
            if quad == 'BL':
                rect = x, dem, w, h
            elif quad == 'TR':
                rect = dem, y, w, h
            elif quad == 'BR':
                rect = dem, dem, w, h
            best[quad] = rect, value

        return best

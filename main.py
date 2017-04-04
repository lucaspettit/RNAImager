# RNA Imager
#
# Creation:
# 12/28/2016
#
# Author(s):
# Lucas Pettit

from tkinter import *
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join, basename

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.image as mping
import matplotlib.patches as patches

from Imager import RemoveToFit


class Application(Frame):
    _debug = True

    _fig = None
    _subplot = None
    _canvas = None
    _title = ''

    _image_dir = ''
    _output_dir = ''
    _paths = []
    _no_image = ''
    _num_images = 0
    _curr_image_index = 0

    _stupid_patches = []
    _curr_image = None

    _imager = None

    def __init__(self, master=None, title='RNA Imager v0.2'):
        # get and load image
        super(Application, self).__init__(master)

        # init junk
        self._title = title
        self._fig = Figure(dpi=100)
        self._subplot = self._fig.add_subplot(111)
        self._image_dir = join('res', 'stock')
        self._output_dir = join('res', 'output')
        self._paths = [f for f in listdir(self._image_dir) if isfile(join(self._image_dir, f))]
        self._no_image = join('res', 'No_Image_Available.png')
        self._num_images = len(self._paths)
        self._curr_image_index = -1
        self._regions = {}

        if self._num_images == 0:
            img = mping.imread(self._no_image)
            master.wm_title = ' - No Image Available'
        else:
            self._curr_image_index = 0
            path = join(self._image_dir, self._paths[self._curr_image_index])
            img = mping.imread(path)
            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

        self._curr_image = img
        self._subplot.imshow(img)
        self._subplot.axis('off')

        # image processing object
        self._imager = RemoveToFit(True)

        # set canvas area
        self._canvas = FigureCanvasTkAgg(self._fig, master=master)
        self._canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self._canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
        self._canvas.callbacks.connect('button_press_event', self.on_click)
        self._canvas.show()

        # make buttons
        self._btn_prev = Button(master=master, text='<- Prev', padx=5, pady=2, command=self._PrevImage)
        self._btn_next = Button(master=master, text='Next ->', padx=5, pady=2, command=self._NextImage)
        self._btn_eval = Button(master=master, text='Eval', padx=10, pady=2, command=self._Eval)
        self._btn_save = Button(master=master, text='Save dat data!', padx=10, pady=2, command=self._saveData)
        self._btn_prev.pack(side=LEFT)
        self._btn_next.pack(side=LEFT)
        self._btn_eval.pack(side=LEFT)
        self._btn_save.pack(side=RIGHT)

    def __del__(self):
        self.quit()

    def _PrevImage(self):
        if self._curr_image_index > 0:
            self._clearRegions()

            self._curr_image_index -= 1
            path = join(self._image_dir, self._paths[self._curr_image_index])
            self._displayImage(mping.imread(path))

            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _NextImage(self):
        if self._curr_image_index < self._num_images - 1:
            self._clearRegions()

            self._curr_image_index += 1
            path = join(self._image_dir, self._paths[self._curr_image_index])
            self._displayImage(mping.imread(path))
            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _Eval(self):
        if self._curr_image_index >= 0:
            image = Image.open(join(self._image_dir, self._paths[self._curr_image_index]))
            self._imager.eval(image, COMPW=720)
            buffer, acc, rej = self._imager.image(viewRaw=False, fullSpectrum=False)

            print('found ' + str(len(acc)))

            regions = {}
            for c, d in acc:
                x, y = c
                if x not in regions:
                    regions[x] = {}
                if y not in regions[x]:
                    regions[x][y] = (d, 0)
            for c, d in rej:
                x, y = c
                if x not in regions:
                    regions[x] = {}
                if y not in regions[x]:
                    regions[x][y] = (d, 1)
            self._regions = regions
            self._displayImage(buffer)

    def _saveData(self):
        dir = join('res', join('training', 'snippets'))
        labels = {}
        saved = {}

        for f in listdir(dir):
            if isfile(join(dir, f)):
                name = f.split('.')[0]
                label = int(name.split("_")[0])
                num = int(name.split('_')[-1])
                if label not in labels or labels[label] < num:
                    labels[label] = num

        for x in self._regions.keys():
            for y in self._regions[x].keys():
                size, label = self._regions[x][y]
                if label not in saved:
                    saved[label] = 0

                if label not in labels:
                    labels[label] = 0

                w, h = size
                snippet = self._curr_image.crop((x, y, x+w, y+h))
                name = str(label) + '_' + str(labels[label]) + '.jpeg'
                snippet.save(join(dir, name), "JPEG")
                labels[label] += 1
                saved[label] += 1

        for key in saved.keys():
            print(str(key) + ' -> ' + str(saved[key]))
        print('Done!')

    def _displayImage(self, image):
        self._curr_image = image
        self._subplot.imshow(self._curr_image)
        self._subplot.axis('off')
        self._drawRegions(self._regions)
        self._canvas.show()

    def _drawRegions(self, regions):
        self._clearRegions()
        self._regions = regions
        for x in regions.keys():
            for y in regions[x].keys():
                size, label = regions[x][y]
                color = 'blue' if label == 0 else 'red'
                p = patches.Rectangle((x, y), size[0], size[1], fill=False, edgecolor=color)
                self._stupid_patches.append(p)
                self._subplot.add_patch(p)

    def _clearRegions(self):
        for p in self._stupid_patches:
            p.remove()
        self._stupid_patches = []
        self._regions = {}

    def on_click(self, event):
        if event.inaxes is not None and len(self._regions) > 0:
            x, y = int(event.xdata), int(event.ydata)
            self._change_label((x, y))
            self._drawRegions(self._regions)

    def _change_label(self, pos):
        cx, cy = pos

        if not isinstance(self._regions, dict):
            return
        for x in self._regions.keys():
            if cx >= x:
                for y in self._regions[x].keys():
                    if cy >= y:
                        size, label = self._regions[x][y]
                        w, h = size
                        if cx <= x + w and cy <= y + h:
                            label = 0 if label == 1 else 1
                            self._regions[x][y] = size, label


root = Tk()
root.title('RNA Imager v0.2')
app = Application(master=root)
app.mainloop()

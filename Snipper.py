# Snipper
#
# Createion:
# 3/17/17
#
# Author:
# Lucas Pettit

from tkinter import *
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join, basename

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.patches as patches

from Imager import *


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
        self._image_dir = join(join('res', 'training'), 'snippets')
        self._output_dir = join('res', 'training')
        self._paths = [f for f in listdir(self._image_dir) if isfile(join(self._image_dir, f))]
        self._no_image = join('res', 'No_Image_Available.png')
        self._num_images = len(self._paths)
        self._curr_image_index = -1
        self._sections = {}

        if self._num_images == 0:
            img = mpimg.imread(self._no_image)
            master.wm_title = ' - No Image Available'
        else:
            self._curr_image_index = 0
            path = join(self._image_dir, self._paths[self._curr_image_index])
            img = mpimg.imread(path)
            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

        self._curr_image = img
        self._subplot.imshow(img)
        self._subplot.axis('off')

        # image processing object
        self._imager = BestFit()

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
            self._displayImage(mpimg.imread(path))

            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _NextImage(self):
        if self._curr_image_index < self._num_images - 1:
            self._clearRegions()

            self._curr_image_index += 1
            path = join(self._image_dir, self._paths[self._curr_image_index])
            self._displayImage(mpimg.imread(path))
            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _Eval(self):
        if self._curr_image_index >= 0:
            image = Image.open(join(self._image_dir, self._paths[self._curr_image_index]))
            self._sections = self._imager.eval(image)
            self._drawRegions(self._sections)
            tot = 0
            for s, v in self._sections.values():
                tot += v
            print("confidence = {0}".format(round(float(tot/4), 4)))

    def _saveData(self):
        pass

    def _displayImage(self, image):
        self._curr_image = image
        self._subplot.imshow(self._curr_image)
        self._subplot.axis('off')
        self._drawRegions(self._sections)

    def _drawRegions(self, sections):
        self._clearRegions()
        self._sections = sections
        conf = 0
        for quadrant in sections.keys():
            rect, value = sections[quadrant]
            conf += value
            x, y, w, h = rect
            if quadrant == 'TL':
                color = 'red'
            elif quadrant == 'TR':
                color = 'magenta'
            elif quadrant == 'BL':
                color = 'blue'
            elif quadrant == 'BR':
                color = 'yellow'
            else:
                color = 'purple'
            p = patches.Rectangle((x, y), w, h, fill=False, edgecolor=color)
            self._stupid_patches.append(p)
            self._subplot.add_patch(p)
        self._canvas.show()

    def _clearRegions(self):
        for p in self._stupid_patches:
            p.remove()
        self._stupid_patches = []
        self._sections = {}

    def on_click(self, event):
        if event.inaxes is not None and len(self._sections) > 0:
            x, y = int(event.xdata), int(event.ydata)
            self._change_label((x, y))
            self._drawRegions(self._sections)

    def _change_label(self, pos):
        pass

root = Tk()
root.title('Sniper v0.1')
app = Application(master=root)
app.mainloop()
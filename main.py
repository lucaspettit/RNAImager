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

        if self._num_images == 0:
            img = mping.imread(self._no_image)
            master.wm_title = ' - No Image Available'
        else:
            self._curr_image_index = 0
            path = join(self._image_dir, self._paths[self._curr_image_index])
            img = mping.imread(path)
            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

        self._subplot.imshow(img)
        self._subplot.axis('off')

        # image processing object
        self._imager = RemoveToFit(True)

        # set canvas area
        self._canvas = FigureCanvasTkAgg(self._fig, master=master)
        self._canvas.show()
        self._canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self._canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # make buttons
        self._btn_prev = Button(master=master, text='<- Prev', padx=5, pady=2, command=self._PrevImage)
        self._btn_next = Button(master=master, text='Next ->', padx=5, pady=2, command=self._NextImage)
        self._btn_eval = Button(master=master, text='Eval', padx=10, pady=2, command=self._Eval)
        self._btn_prev.pack(side=LEFT)
        self._btn_next.pack(side=LEFT)
        self._btn_eval.pack(side=LEFT)

    def __del__(self):
        self.quit()

    def _PrevImage(self):
        if self._curr_image_index > 0:
            self._curr_image_index -= 1
            path = join(self._image_dir, self._paths[self._curr_image_index])
            self._subplot.imshow(mping.imread(path))
            self._subplot.axis('off')

            for p in self._stupid_patches:
                p.remove()
            self._stupid_patches = []

            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _NextImage(self):
        if self._curr_image_index < self._num_images - 1:
            self._curr_image_index += 1
            path = join(self._image_dir, self._paths[self._curr_image_index])
            self._subplot.imshow(mping.imread(path))
            self._subplot.axis('off')

            for p in self._stupid_patches:
                p.remove()
            self._stupid_patches = []

            self._canvas.show()

            if self._debug:
                print('image -> ' + str(basename(path).split('.')[0]))

    def _Eval(self):
        if self._curr_image_index >= 0:
            image = Image.open(join(self._image_dir, self._paths[self._curr_image_index]))
            self._imager.eval(image, COMPW=256)
            buffer, regions = self._imager.image()
            self._subplot.imshow(buffer, interpolation='nearest')
            self._subplot.axis('off')

            for r in regions:
                coord, size = r
                p = patches.Rectangle(coord, size[0], size[1], fill=False, edgecolor='blue')
                self._stupid_patches.append(p)

                self._subplot.add_patch(p)

            self._canvas.show()


root = Tk()
root.title('RNA Imager v0.2')
app = Application(master=root)
app.mainloop()

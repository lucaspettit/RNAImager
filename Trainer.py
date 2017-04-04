# Trainer
#
# Createion:
# 3/23/17
#
# Author:
# Lucas Pettit

from tkinter import *
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join, basename
from random import randint

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
    _curr_x = numpy.zeros(784)
    _curr_y = 0

    _imager = None

    def __init__(self, master=None, title='RNA Imager v0.2'):
        # get and load image
        super(Application, self).__init__(master)

        # init junk
        self._title = title
        self._fig = Figure(dpi=100)
        self._subplot = self._fig.add_subplot(111)
        self._image_dir = join(join('res', 'training'), 'corners')
        self._output_dir = join('res', 'training')
        self._paths = [f for f in listdir(self._image_dir) if isfile(join(self._image_dir, f))]
        self._no_image = join('res', 'No_Image_Available.png')
        self._num_images = len(self._paths)
        self._curr_image_index = -1
        self._sections = {}

        # image processing object
        self._imager = BestFit('')

        # set canvas area
        self._canvas = FigureCanvasTkAgg(self._fig, master=master)
        self._canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self._canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # make buttons
        self._btn_next = Button(master=master, text='Next ->', padx=5, pady=2, command=self._NextImage)
        self._btn_save = Button(master=master, text='Save', padx=10, pady=2, command=self._save)
        self._btn_next.pack(side=LEFT)
        self._btn_save.pack(side=RIGHT)

        if self._num_images == 0:
            img = mpimg.imread(self._no_image)
            master.wm_title = ' - No Image Available'
        else:
            self._curr_image_index = 0
            self.showImage()

    def __del__(self):
        self.quit()

    def _NextImage(self):
        if self._curr_image_index == self._num_images - 1:
            self._curr_image_index = 0

        self._curr_image_index += 1
        self.showImage()

    def showImage(self):
        missed = False
        path = join(self._image_dir, self._paths[self._curr_image_index])
        self._curr_y = int(self._paths[self._curr_image_index].split('_')[-1].split('.')[0])

        image = mpimg.imread(path)
        self._displayImage(image)
        self._canvas.show()

        res = self._Eval()

        if res != self._curr_y:
            missed = True
            if self._curr_y == 0:
                self._imager.train(self._curr_x, -1)
            else:
                self._imager.train(self._curr_x, 1)

        if self._debug:
            img_num = int(basename(path).split('.')[0].split('_')[0])
            img_name = str(img_num) + ' ' if img_num < 10 else str(img_num)
            print('image {0} : {1} -> {2} {3}'.format(img_name, self._curr_y, res, 'miss' if missed else ''))

    def _save(self):
        path = join('res', 'w.dat')
        f = open(path, 'w')
        s = ''
        w = self._imager.w()
        for i in range(len(w)):
            s += str(w[i]) + ' '
        f.write(s)
        f.close()

    def _Eval(self):
        if self._curr_image_index >= 0:
            image = Image.open(join(self._image_dir, self._paths[self._curr_image_index]))
            self._curr_x = numpy.array(image.getdata(band=1))
            conf = self._imager.predictRT(self._curr_x)
            if conf > 0:
                return 1
            if conf < 0:
                return 0
            else:
                return -1

    def _displayImage(self, image):
        self._curr_image = image
        self._subplot.imshow(self._curr_image)
        self._subplot.axis('off')
        self._canvas.show()

root = Tk()
root.title('Trainer v0.1')
app = Application(master=root)
app.mainloop()
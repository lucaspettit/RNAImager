
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join, basename

directory = 'corners'
paths = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
outfile = open('data.dat', 'w')

for f in paths:
    img = Image.open(f)
    w, h = img.size
    scailer = 28/max(w, h)
    w *= scailer
    h *= scailer
    label = 1 if int(basename(f).split('_')[-1].split('.')[0]) > 0 else -1

    img = img.resize((28, 28))
    data = img.getdata(band=1)
    feature = ''
    for d in data:
        feature += str(d) + ' '
    feature += str(label) + '\n'
    outfile.write(feature)

outfile.close()

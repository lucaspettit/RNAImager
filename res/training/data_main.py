
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join, basename

paths = [join('snippets', f) for f in listdir('snippets') if isfile(join('snippets', f))]
outfile = open('data.dat', 'w')

for f in paths:
    img = Image.open(f)
    w, h = img.size
    scailer = 28/max(w, h)
    w *= scailer
    h *= scailer
    squariness = abs((w - h)/max(w, h))
    label = basename(f).split('_')[0]

    img = img.resize((28, 28))
    data = img.getdata(band=1)
    feature = str(int(w)) + ' ' + str(int(h)) + ' ' + str(squariness) + ' '
    for d in data:
        feature += str(d) + ' '
    feature += str(label) + '\n'
    outfile.write(feature)

outfile.close()

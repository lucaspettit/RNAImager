import numpy as np
import matplotlib.pyplot as mp
import matplotlib.image as immp
import pygame
import tkinter


_root_img_path = '../res/stock/'
_img_name = 'img3_.tif'


def neighbors(x, y, w, h):
    n = []
    if x-1 >= 0:
        n.append((x-1, y))
    if x+1 < w:
        n.append((x+1, y))
    if y-1 >= 0:
        n.append((x, y-1))
    if y+1 < h:
        n.append((x, y+1))
    if x-1 >= 0 and y-1 >= 0:
        n.append((x-1, y-1))
    if x-1 >= 0 and y+1 < h:
        n.append((x-1, y + 1))
    if x+1 < w and y-1 >= 0:
        n.append((x+1, y-1))
    if x+1 < w and y+1 < h:
        n.append((x+1, y+1))

    return n


def deltamatrix(pixels, dementions):

    w, h = dementions
    d = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            delta = 0
            n = neighbors(x, y, w, h)
            for nx, ny in n:
                delta += int(pixels[x, y]) - int(pixels[nx, ny])
            d[x, y] = delta / len(n)
    return d

# --- Load, compress, and translate image to BW and MC --- #
_img = pygame.image.load(_root_img_path + _img_name)
w, h = _img.get_rect().size
jw, jh = int(w/256), int(h/256)

pixels = pygame.surfarray.array3d(_img)
pixels = pixels[::jw, ::jh, :]

w, h = int(w/jw), int(h/jh)

orig = pixels
mc = pixels[:, :, 1]
bw = np.zeros((w, h))
for x in range(w):
    for y in range(h):
        r, g, b = pixels[x, y]
        pixel = (int(r)+int(g)+int(b))/3
        bw[x, y] = int(pixel)


# get delta picture
delta = deltamatrix(mc, (w, h))


mc = np.transpose(mc)
delta = np.transpose(delta)

# get statistics
mc_mean = np.mean(mc)
mc_std = np.std(mc)
delta_mean = np.mean(delta)
delta_std = np.std(delta)

mp.subplot(221)
mp.imshow(mc, cmap='gray', interpolation='nearest')

mp.subplot(222)
mp.imshow(delta)



# --- Get Pixel Color Graph --- #
mp.subplot2grid((4, 2), (2, 4))
mp.xlabel('pixel color')
mp.ylabel('number of pixels')
mp.axis([-50, 50, 0, 15000])
#mp.hist(mc.ravel(), bins=256, fc='g', alpha=0.5, label='full')
mp.hist(delta.ravel(), bins=256, fc='b', alpha=0.5, label='b&w')

for x in range(-4, 4):
    mp.axvline(delta_mean + (x * delta_std), color='g', linestyle='dashed')
mp.axvline(delta_mean, color='r', linestyle='dashed')


mp.show()




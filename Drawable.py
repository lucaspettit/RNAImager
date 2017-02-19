import pygame
import os


class Direction(object):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (-1, 0)
    WEST = (1, 0)
    NORTHEAST = (-1, -1)
    NORTHWEST = (1, -1)
    SOUTHEAST = (-1, 1)
    SOUTHWEST = (1, 1)
    _dir = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]

    def __contains__(self, item):
        return item in self._dir


class Drawable(object):
    _rect = (0,0,0,0)
    _working_rect = (0,0,0,0)
    _padding = (2, 2, 2, 2)
    _background_color = (255, 255, 255)
    _background_color_clicked = (255, 255, 255)
    _background_color_disabled = (110, 110, 110)
    _boarder_color = (0,0,0)
    _boarder_thickness = 2
    _canvas = 0
    _clicked = False

    enable = True

    def __init__(self, canvas, rect):
        self._rect = rect
        self._canvas = canvas
        self._init_layout_()

    def _init_layout_(self):
        x, y, w, h = self._rect
        x += self._padding[0]
        y += self._padding[1]
        w -= (self._padding[0] + self._padding[2])
        h -= (self._padding[1] + self._padding[3])
        self._rect = (x, y, w, h)

        x += self._boarder_thickness
        y += self._boarder_thickness
        w -= (2*self._boarder_thickness)
        h -= (2*self._boarder_thickness)
        self._working_rect = (x, y, w, h)

    def draw(self):
        if self._clicked:
            pygame.draw.rect(self._canvas, self._background_color_clicked, self._rect, 0)
        elif not self.enable:
            pygame.draw.rect(self._canvas, self._background_color_disabled, self._rect, 0)
        else:
            pygame.draw.rect(self._canvas, self._background_color, self._rect, 0)

        pygame.draw.rect(self._canvas, self._boarder_color, self._rect, self._boarder_thickness)

    def clicked(self, pos, btn):
        if not self.enable:
            return

        x, y = pos
        if x in range(self.x(), self.x()+self.width()):
            if y in range(self.y(), self.y()+self.height()):
                self._clicked = True
                return True
        self._clicked = False
        return False

    def unclick(self):
        self._clicked = False
        self.draw()

    def backgroundColor(self, color = None):
        if color is None:
            return self._background_color
        else:
            self._background_color = color
            bgc = self._background_color
            r = int(bgc[0] * 0.9)
            g = int(bgc[1] * 0.9)
            b = int(bgc[2] * 0.9)
            self._background_color_clicked = (r, g, b, 30)
            return color

    def boarderColor(self, color = None):
        if color is None:
            return self._boarder_color
        else:
            self._boarder_color = color

    def x(self, value = None):
        if value is None:
            return self._rect[0]
        else:
            self._rect[0] = value

    def y(self, value = None):
        if value is None:
            return self._rect[1]
        else:
            self._rect[1] = value

    def width(self, value = None):
        if value is None:
            return self._rect[2]
        else:
            self._rect[2] = value

    def height(self, value = None):
        if value is None:
            return self._rect[3]
        else:
            self._rect[3] = value

    def workingRect(self, value = None):
        if value is None:
            return self._working_rect
        else:
            self._working_rect = value

    def workingX(self):
        return self._working_rect[0]

    def workingY(self):
        return self._working_rect[1]

    def workingWidth(self):
        return self._working_rect[2]

    def workingHeight(self):
        return self._working_rect[3]

    def padding(self, value = None):
        if value is None:
            return self._padding
        else:
            self._padding = value
            self._init_layout_()


class Button(Drawable):
    _font = 0
    _font_x, _font_y = 0, 0
    _text = "button"
    _font_color = (0,0,0)
    _font_family = "Arial"
    _font_size = 20

    def __init__(self, canvas, rect, text):
        super(Button, self).__init__(canvas, rect)
        pygame.font.init()
        self._font = pygame.font.SysFont(self._font_family, self._font_size)
        self._text = text
        self._init_text_()

        bgc = self._background_color
        r = int(bgc[0] * 0.75)
        g = int(bgc[1] * 0.75)
        b = int(bgc[2] * 0.75)
        self._background_color_clicked = (r, g, b)
        r = int(bgc[0] * 0.40)
        g = int(bgc[1] * 0.40)
        b = int(bgc[2] * 0.40)
        self._background_color_disabled = (r, g, b)

    def _init_text_(self):
        tw, th = self._font.size(self._text)
        x, y, w, h = self.workingRect()
        self._font_x = x + ((w - tw) / 2)
        self._font_y = y + ((h - th) / 2)

    def draw(self):
        super(Button, self).draw()
        surface = self._font.render(self._text, True, self._font_color)
        self._canvas.blit(surface, (self._font_x, self._font_y))


class ImageBox(Drawable):
    _raw_image = None
    _raw_image_size = (0, 0)
    _viewable_image = None
    _viewable_image_size = (0, 0)
    _img_ratio = (0, 0)
    _img_path = ""
    _img_rect = None
    _zoom = 0

    def __init__(self, canvas, rect, path = None):
        super(ImageBox, self).__init__(canvas, rect)
        if path is not None:
            dir = os.path.dirname(os.path.abspath(__file__))
            self._img_path = dir + "\\" + path
            self._init_image_()

    def _init_image_(self, obj):
        try:
            self._raw_image = pygame.image.load(os.path.join(obj))
            self._raw_image_size = self._raw_image.get_rect().size
            self._img_path = obj
            self._render_image_()
        except pygame.error:
            print("Unable to open image")

    def _render_image_(self):
        x, y, w, h = self.workingRect()
        self._viewable_image = pygame.transform.scale(self._raw_image, (w, h))
        self._viewable_image_size = self._viewable_image.get_rect().size
        self._img_rect = (x, y, w, h)

        ratioX = float(self._viewable_image_size[0]) / float(self._raw_image_size[0])
        ratioY = float(self._viewable_image_size[1]) / float(self._raw_image_size[1])
        self._img_ratio = (ratioX, ratioY)

    def image_from_path(self, obj):
        if obj is not None:
            self._init_image_(obj)
        else:
            return self._raw_image

    def image(self, img):
        if img is not None:
            self._raw_image = img
            self._raw_image_size = self._raw_image.get_rect().size
            self._render_image_()

    def save(self, path):
        try:
            pygame.image.save(self._raw_image, path)
        except IndexError:
            print('Unable to save image')

    def draw(self):
        super(ImageBox, self).draw()
        if self._viewable_image is not None:
            self._canvas.blit(self._viewable_image, self._img_rect)

    def getClickPoint(self, pos):
        if self._raw_image is None:
            return -1, -1

        x, y = (pos[0] - self.workingX(), pos[1] - self.workingY())
        rx, ry = self._img_ratio
        x, y = (float(x)/rx, float(y)/ry)
        return int(x), int(y)

    def zoom(self, amount):
        self._zoom = amount

    def pan(self, dir):
        if dir == Direction.NORTH:
            x = 0
        elif dir == Direction.SOUTH:
            x = 0


class PixelBox(Drawable):
    _font = 0
    _font_x, _font_y = 0, 0
    _text = "pixel box"
    _font_color = (0, 0, 0)
    _font_family = "Arial"
    _font_size = 20
    _pixel_default_color = (255,255,255)
    _pixel_box = (0, 0, 0, 0)
    _pixel_color = _pixel_default_color
    _pixel_box_border_color = (110, 110, 110)

    def __init__(self, canvas, rect, text):
        super(PixelBox, self).__init__(canvas, rect)
        self._boarder_color = self._pixel_default_color
        self._background_color_disabled = self._background_color
        pygame.font.init()
        self._font = pygame.font.SysFont(self._font_family, self._font_size)
        self._text = text
        x, y, w, h = rect
        px, py, pw, ph = self._padding
        self._pixel_box = (x + w - h, y + py, h - ph, h-ph)

        tw, th = self._font.size(self._text)
        x, y, w, h = self.workingRect()
        self._font_x = x + ((w - tw) / 2)
        self._font_y = y + ((h - th) / 2)

    def setColor(self, c):
        self._pixel_color = c, c, c

    def draw(self):
        super(PixelBox, self).draw()
        pygame.draw.rect(self._canvas, self._pixel_color, self._pixel_box, 0)
        pygame.draw.rect(self._canvas, self._pixel_box_border_color, self._pixel_box, self._boarder_thickness)
        surface = self._font.render(self._text, True, self._font_color)
        self._canvas.blit(surface, (self._font_x, self._font_y))

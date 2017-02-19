import pygame

class ImageBox:
    _rect = 0
    _working_rect = 0
    _background_color = (255, 255, 255)
    _boarder_color = (0,0,0)
    _boarder_thickness = 2
    _canvas = 0

    def __init__(self, canvas, rect):
        self._rect = rect
        x, y, w, h = rect
        x -= self._boarder_thickness
        y -= self._boarder_thickness
        w -= self._boarder_thickness
        h -= self._boarder_thickness
        x /= 2
        y /= 2
        self._working_rect = (x, y, w, h)
        self._canvas = canvas

    def setBackgroundColor(self, color):
        self._background_color = color

    def setBoarderColor(self, color):
        self._boarder_color = color

    def draw(self):
        pygame.draw.rect(self._canvas, self._background_color, self._rect, 0)
        pygame.draw.rect(self._canvas, self._boarder_color, self._rect, self._boarder_thickness)

    def X(self):
        return self._rect[0]

    def Y(self):
        return self._rect[1]

    def Width(self):
        return self._rect[2]

    def Height(self):
        return self._rect[3]

    def WorkingRect(self):
        return self._working_rect

from PIL import Image, ImageDraw


class TileCanvasGenerator():
    def __init__(self, img_x, img_y):
        self.img_x = img_x
        self.img_y = img_y
        self.size = (self.img_x, self.img_y)

    def make_empty_tile(self, color=None):
        if color:
            return Image.new('RGB', self.size, color)
        else:
            return Image.new('RGB', self.size)

    def make_draw_canvas_tile(self):
        img = Image.new('RGB', self.size)
        return ImageDraw.Draw(img)

tg = TileCanvasGenerator(64, 64)
print(tg.make_empty_tile())
print(tg.make_draw_canvas_tile())

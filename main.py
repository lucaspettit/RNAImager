# RNA Imager
#
# Creation:
# 12/28/2016
#
# Author(s):
# Lucas Pettit

from Drawable import *
from Imager import RemoveToFit
from os import listdir
from os.path import isfile, join, basename

#
# Create/initialize local variables
#
# Window variables
background_color = (255, 255, 255)
width, height = 600, 650
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('RNA Imager')
gui_objects = []

# path variables
image_dir = 'res/stock'
output_dir = 'res/output'
no_image_available_path = 'res/No_Image_Available.png'
res_image_paths = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
num_res_images = len(res_image_paths)
curr_image_index = 0

# image processing object
#imager = ImageUtil(None)
imager = RemoveToFit(True)
imager.compressionRatio(256)

#
#  create buttons
#
# btn_next
rect = (0, 0, 110, 40)
btn_next = Button(window, rect, "Next Image")
if num_res_images <= 0:
    btn_next.enable = False
gui_objects.append(btn_next)

# btn_eval
rect = (115, 0, 90, 40)
btn_eval = Button(window, rect, "Evaluate")
btn_eval.enable = False
gui_objects.append(btn_eval)

# btn_reset
rect = (210, 0, 90, 40)
btn_reset = Button(window, rect, "Reset")
btn_reset.enable = False
gui_objects.append(btn_reset)

# btn_step
rect = (305, 0, 90, 40)
btn_step = Button(window, rect, "Step")
btn_step.enable = False
gui_objects.append(btn_step)

#
# threshold box
#
rect = (width-110, 0, 110, 40)
thresholdBox = PixelBox(window, rect, "Th: ")
thresholdBox.enable = False
gui_objects.append(thresholdBox)

#
# create image box
#
rect = (0, rect[1]+rect[3], width, height-rect[1]-rect[3])
canvas = ImageBox(window, rect)
canvas.backgroundColor((0,0,0))
canvas.image_from_path(no_image_available_path)
gui_objects.append(canvas)

#
# initial draw for window
#
window.fill(background_color)
for obj in gui_objects:
    obj.draw()
pygame.display.flip()

#
# GUI event loop
#
running = True
while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT: # quit
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN: # mouse button click
            if btn_next.clicked(event.pos, event.button): # btn_next clicked

                # this shouldn't happen but just in case...
                if curr_image_index >= num_res_images:
                    btn_next.enables = False
                    btn_eval.enable = False
                    btn_reset.enable = False
                    btn_step.enables = False
                    canvas.image_from_path(no_image_available_path)

                # this should happen
                else:
                    #imager.setImage(join(image_dir, res_image_paths[curr_image_index]))
                    #canvas.image(imager.getImage())

                    imager.image(join(image_dir, res_image_paths[curr_image_index]))
                    canvas.image(imager.image())
                    btn_eval.enable = True
                    btn_reset.enable = False
                    btn_step.enable = False
                    curr_image_index += 1

                    # this if statement is why the first check shouldn't happen
                    if curr_image_index >= num_res_images:
                        btn_next.enable = False

            elif btn_eval.clicked(event.pos, event.button): # btn_eval clicked
                btn_eval.draw()
                pygame.display.flip()

                #imager.findBrightSpots()
                #canvas.image(imager.getImage())
                imager.eval()
                canvas.image(imager.image())
                btn_reset.enable = True
                btn_eval.enable = False
                btn_step.enable = True
                #thresholdBox.setColor(imager.getThreshold())
                canvas.draw()

            elif btn_reset.clicked(event.pos, event.button): # btn_reset clicked
                btn_reset.draw()
                pygame.display.flip()

                imager.reset()
                btn_eval.enable = True
                btn_step.enable = False
                btn_reset.enable = False
                #thresholdBox.setColor(imager.getThreshold())
                #canvas.image(imager.getImage())
                canvas.image(imager.image())
                canvas.draw()

            elif btn_step.clicked(event.pos, event.button): # btn_step clicked
                btn_step.draw()
                imager.step()
                canvas.image(imager.image())
                #imager.saveSelected()
                #imager.save(join(output_dir, basename(res_image_paths[curr_image_index-1]).split(".")[0] + '.png'))
                #if curr_image_index == num_res_images-1:
                #    imager.saveData()

            #elif canvas.clicked(event.pos, event.button): # canvas clicked
                #_x, _y = canvas.getClickPoint(event.pos)
                #if imager.selectPos((_x, _y), event.button):
                #    btn_step.enable = True
                #pixel = imager.getPixelAt(event.pos)
                #if pixel is not None:
                #    r, g, b = pixel
                #    pixel = int((float(r)+float(g)+float(b)) / 3.0)
                    #imager.setThreshold(pixel)
                    #thresholdBox.setColor(imager.getThreshold())
                #canvas.image(imager.getImage())
                #canvas.draw()


        elif event.type == pygame.MOUSEBUTTONUP:
            for obj in gui_objects:
                obj.unclick()

        pygame.display.flip()

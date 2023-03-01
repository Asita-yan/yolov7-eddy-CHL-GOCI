import pickle

import numpy as np
from PIL import ImageDraw, ImageFont,Image
import colorsys
with open('E:\\photo\\predict\\1.pkl','rb') as f:
    day_results=pickle.load(f)



top_label=day_results
file = 'E:\\photo\\00\\20110401.jpg'
image = Image.open(file).convert('RGB')

# font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
# thickness = int(max((image.size[0] + image.size[1]) // np.mean(image.size[0] , image.size[1]), 1))
hsv_tuples = [(x / 2, 1., 1.) for x in range(2)]
# colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
# colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))



draw = ImageDraw.Draw(image)

# image.show()

for i in range(24):
    draw.rectangle([int(top_label[i,1]), int(top_label[i,0]), int(top_label[i,3]), int(top_label[i,2])], width=5, outline='red')

    # draw.rectangle([tuple(20), tuple(30)],fill='red')


image.save('E:\\photo\\predict\\1.jpg')
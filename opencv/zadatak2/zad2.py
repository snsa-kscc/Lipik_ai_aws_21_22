import os
import glob
from PIL import Image

imgPath = "images"
files = glob.glob(imgPath + '/**/*.jpg', recursive=True)

# TODO ispisite sve putanje

path = "/Users/snsa_kscc/Documents/lira/beautiful-soup-env/opencv/zadatak2"

for file in files:
    print(file)

# TODO svaku sliku pretvorite u grayscale i spremite u output direktorij pod nazivom img_x.jpg pri cemu je x redni broj slike

    img = Image.open(file)
    img_gray = img.convert('L')
    img_gray.save(os.path.join(path, 'output', 'img_' +
                               str(files.index(file)) + '.jpg'))

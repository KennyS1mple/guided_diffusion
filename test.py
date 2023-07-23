# CACIOUS CODING
# Data     : 7/23/23  3:37 PM
# File name: test.py
# Desc     :

from PIL import Image

img = Image.open("datasets/luggage.jpg")
img.resize((32, 64), resample=Image.BOX).show("1")
img.resize((64, 64)).show("2")
img.show("self")

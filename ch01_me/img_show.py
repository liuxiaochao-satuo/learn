import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread(r'dataset\lena_gray.png')
plt.imshow(img)
plt.show()
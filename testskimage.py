from skimage import io
from skimage import feature
import matplotlib.pyplot as plt

im = io.imread('Wrist.jpg')
edges1 = feature.canny(im, sigma = 0.05)
edges2 = feature.canny(im, sigma = 0.5)
edges3 = feature.canny(im, sigma = 1.0)
edges4 = feature.canny(im, sigma = 1.5)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

ax1.imshow(edges1, cmap=plt.cm.gray)
ax1.set_title('Canny filter, $\sigma=0.05$', fontsize=10)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.set_title('Canny filter, $\sigma=0.5$', fontsize=10)

ax3.imshow(edges3, cmap=plt.cm.gray)
ax3.set_title('Canny filter, $\sigma=1.0$', fontsize=10)

ax4.imshow(edges4, cmap=plt.cm.gray)
ax4.set_title('Canny filter, $\sigma=1.5$', fontsize=10)


fig.tight_layout()

plt.show()

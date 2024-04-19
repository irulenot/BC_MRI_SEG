import numpy as np
import matplotlib.pyplot as plt

# Load your three images here
image0 = plt.imread('outputs/[\'dm\']_5.png')
image1 = plt.imread('outputs/[\'ispy1\']_3.png')
image2 = plt.imread('outputs/[\'rider\']_1.png')
image3 = plt.imread('outputs/[\'duke\']_20.png')
image0 = image0[:, 150:-150, :]
image1 = image1[:, 150:-150, :]
image2 = image2[:, 150:-150, :]
image3 = image3[:, 150:-150, :]

# Combine the images vertically
combined_image = np.concatenate([image0, image1, image2, image3], axis=0)

# Display the combined image
plt.imshow(combined_image)
plt.axis('off')  # Turn off axis
plt.imsave('outputs/combined_image.png', combined_image)
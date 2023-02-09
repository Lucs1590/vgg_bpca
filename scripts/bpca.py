from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from im2col import gen_sliding_windows, make_blocks, revert_block
import matplotlib.pyplot as plt

def apply_bpca():
    # Generate blocks from the image
    _, blocks, coords = make_blocks(X, window_height=3, window_width=3)

    # Reshape the blocks into 2D arrays
    X_blocks = np.array([block.reshape((-1)) for block in blocks])

    # Scale the values in the blocks
    scaler = MinMaxScaler()
    X_rescaled = scaler.fit_transform(X_blocks)

    # Apply PCA to the rescaled blocks
    pca = PCA(n_components=0.95)
    reduced_blocks = pca.fit_transform(X_rescaled)

    return reduced_blocks, coords, pca, scaler

def revert_bpca(reduced_blocks, coords, original_shape):
    # Revert the PCA transformation
    X_reverted = pca.inverse_transform(reduced_blocks)

    # Revert the scaling
    X_reverted = scaler.inverse_transform(X_reverted)

    # Reshape the reverted blocks back into 3D arrays
    X_reverted = np.array([block.reshape((3, 3)) for block in X_reverted])

    # Revert the blocks back into an image
    reverted_img = revert_block(X_reverted, coords, original_shape)

    return reverted_img

# Load the image and convert it to grayscale
img = Image.open("example.png").convert("L")

# Convert the image to a numpy array
X = np.array(img)

# Save the original image
plt.imshow(X, cmap="Greys")
plt.savefig("X.jpeg")

bpca_img, coords, pca, scaler = apply_bpca()
plt.imshow(bpca_img, cmap="Greys")
reversed_img = revert_bpca(bpca_img, coords, X.shape)
plt.imshow(reversed_img, cmap="Greys")
plt.savefig("reverted.jpeg")
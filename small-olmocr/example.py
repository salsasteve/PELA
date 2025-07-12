import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Tuple

def load_image_grayscale(filepath: str) -> Image:
    """
    Loads an image from a file and converts it to grayscale.

    Args:
        filepath: The path to the image file.

    Returns:
        A PIL Image object in grayscale ('LA') mode.
    """
    print(f"Loading and converting '{filepath}' to grayscale...")
    img = Image.open(filepath)
    return img.convert('LA')

def image_to_matrix(image: Image) -> np.matrix:
    """
    Converts a PIL image object into a NumPy matrix.

    Args:
        image: A PIL Image object.

    Returns:
        A NumPy matrix representing the image's luminance data.
    """
    print("Converting image to matrix...")
    img_array = np.array(list(image.getdata(band=0)), float)
    img_array.shape = (image.size[1], image.size[0])
    return np.matrix(img_array)

def perform_svd(matrix: np.matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Singular Value Decomposition on a matrix.

    Args:
        matrix: The input NumPy matrix.

    Returns:
        A tuple containing the U, sigma, and V components.
    """
    print("Performing SVD...")
    U, sigma, V = np.linalg.svd(matrix)
    print(f"  - U shape: {U.shape}\n  - Sigma shape: {sigma.shape}\n  - V shape: {V.shape}")
    return U, sigma, V

def reconstruct_from_svd(U: np.ndarray, sigma: np.ndarray, V: np.ndarray, rank: int) -> np.matrix:
    """
    Reconstructs a matrix from its SVD components using a specified rank.

    Args:
        U: The U component from SVD.
        sigma: The sigma component (singular values) from SVD.
        V: The V component from SVD.
        rank: The number of singular values to use for reconstruction.

    Returns:
        The reconstructed NumPy matrix of the given rank.
    """
    return np.matrix(U[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(V[:rank, :])

def display_matrix_as_image(matrix: np.matrix, title: str = ""):
    """
    Displays a NumPy matrix as a grayscale image using matplotlib.

    Args:
        matrix: The matrix to display.
        title: The title for the plot.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='gray')
    plt.title(title)
    plt.show()

def main():
    """
    Orchestrates the SVD image compression process using modular functions.
    """
    image_path = 'panda.jpg'
    ranks_to_show = [1, 5, 10, 20, 30, 50]

    try:
        # Step 1: Load and display the original image
        img_gray = load_image_grayscale(image_path)
        display_matrix_as_image(img_gray, "Original Grayscale Image")

        # Step 2: Convert to a matrix
        img_matrix = image_to_matrix(img_gray)
        print(f"Original matrix shape: {img_matrix.shape}")

        # Step 3: Perform SVD
        U, sigma, V = perform_svd(img_matrix)

        # Step 4: Reconstruct and show the image at different ranks
        print("\nReconstructing image with increasing ranks...")
        for r in ranks_to_show:
            print(f"  - Reconstructing with r={r}")
            reconstructed_matrix = reconstruct_from_svd(U, sigma, V, r)
            display_matrix_as_image(reconstructed_matrix, f"Reconstructed Image (r = {r})")

    except FileNotFoundError:
        print(f"\nError: '{image_path}' not found.")
        print("Please ensure the image file is in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
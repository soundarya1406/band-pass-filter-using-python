import cv2
import numpy as np

def bandpass_filter(image, low_cutoff, high_cutoff):
    if image is None:
        raise ValueError("Input image is empty")

    # Perform Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create mask for bandpass filtering
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[center_row - high_cutoff:center_row + high_cutoff, center_col - high_cutoff:center_col + high_cutoff] = 1
    mask[center_row - low_cutoff:center_row + low_cutoff, center_col - low_cutoff:center_col + low_cutoff] = 0

    # Apply mask
    f_transform_shifted_filtered = f_transform_shifted * mask

    # Inverse Fourier Transform
    f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
    filtered_image = np.fft.ifft2(f_transform_filtered)
    filtered_image = np.abs(filtered_image)

    return filtered_image.astype(np.uint8)

# Load an image
image = cv2.imread('C:/Users/sound/Downloads/sharktank(10801080).jpg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Define low and high cutoff frequencies
    low_cutoff = 20
    high_cutoff = 50

    try:
        # Apply band-pass filter
        filtered_image = bandpass_filter(image, low_cutoff, high_cutoff)

        # Display the original and filtered images
        cv2.imshow('Original Image', image)
        cv2.imshow('Bandpass Filtered Image', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)

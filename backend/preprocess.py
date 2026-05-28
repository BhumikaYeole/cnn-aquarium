from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert("L")

    # Convert to numpy array
    img_arr = np.array(img)

    # Detect non-white pixels (the drawing)
    mask = img_arr < 250

    # Find bounding box
    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop to drawing
    cropped = img_arr[y_min:y_max+1, x_min:x_max+1]

    # Add small padding
    padding = 10
    cropped = np.pad(
        cropped,
        pad_width=padding,
        mode='constant',
        constant_values=255
    )

    # Resize to 28x28
    resized = Image.fromarray(cropped).resize((28, 28))

    # Convert back to numpy
    final_img = np.array(resized)

    # Invert colors so doodle becomes bright
    final_img = 255 - final_img

    # Normalize
    final_img = final_img.astype("float32") / 255.0

    # Add channel dimension
    final_img = np.expand_dims(final_img, axis=-1)

    # Add batch dimension
    final_img = np.expand_dims(final_img, axis=0)

    return final_img
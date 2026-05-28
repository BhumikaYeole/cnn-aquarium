from PIL import Image
import numpy as np

def preprocess_image(image_path):

    img = Image.open(image_path).convert("RGBA")

    # Convert transparent bg to white
    white_bg = Image.new("RGBA", img.size, (255,255,255,255))
    img = Image.alpha_composite(white_bg, img)

    # Convert to grayscale
    img = img.convert("L")

    img_arr = np.array(img)

    # fish becomes white
    # background becomes black
    img_arr = 255 - img_arr

    # Threshold to remove gray noise
    img_arr[img_arr < 30] = 0

    # Find non-black pixels
    coords = np.argwhere(img_arr > 0)

    if len(coords) == 0:
        # blank image fallback
        return np.zeros((1,28,28,1), dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop tightly around drawing
    cropped = img_arr[y_min:y_max+1, x_min:x_max+1]

    # Resize directly
    resized = Image.fromarray(cropped).resize((20,20))

    # Create clean 28x28 black canvas
    final_canvas = np.zeros((28,28), dtype=np.uint8)

    # Center the resized drawing
    start_x = (28 - 20)//2
    start_y = (28 - 20)//2

    final_canvas[
        start_y:start_y+20,
        start_x:start_x+20
    ] = np.array(resized)

    # Normalize
    final_canvas = final_canvas.astype("float32") / 255.0

    # Add dimensions
    final_canvas = np.expand_dims(final_canvas, axis=-1)
    final_canvas = np.expand_dims(final_canvas, axis=0)

    return final_canvas
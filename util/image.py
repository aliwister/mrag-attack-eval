import base64, io, hashlib, random, math, numpy as np
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO


def resize_base64_image(img, size=(224, 224)):  # CLIP expects 224x224 images
    """Convert PIL Image to base64 string"""
    if not isinstance(img, Image.Image):
        raise TypeError("Input must be a PIL Image")
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    format = img.format if img.format else 'JPEG'
    resized_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def random_crop_area(img, keep_ratio=0.6):
    w, h = img.size
    area = w * h
    
    # target area = keep_ratio * original area
    target_area = area * keep_ratio
    
    # keep same aspect ratio as original
    aspect = w / h
    new_w = int(math.sqrt(target_area * aspect))
    new_h = int(math.sqrt(target_area / aspect))
    
    # pick random top-left corner for crop
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    right = left + new_w
    bottom = top + new_h
    #print(left, top, right, bottom)
    cropped = img.crop((left, top, right, bottom))

    return cropped

def random_rotate_flip(img):
    """Randomly rotate and/or flip an image."""
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)
    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_erasing(img, scale=0.2):
    """Randomly erase a rectangular patch."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    erase_w, erase_h = int(w * scale), int(h * scale)
    x = random.randint(0, w - erase_w)
    y = random.randint(0, h - erase_h)
    arr[y:y+erase_h, x:x+erase_w] = 0  # black patch
    return Image.fromarray(arr)  

def add_gaussian_noise(img, mean=0, std=25):
    """
    Add Gaussian noise to an image.
    mean, std are in pixel intensity units (0–255).
    """
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def post_process_image(img, flag, rag_size=(224,224)):
    if flag == "NONE":
        return img
    elif flag == "RESIZE":
        return img.resize(rag_size)   
    elif flag == "PIXELATE":
        block = 2
        w, h = img.size
        img_small = img.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
        return img_small.resize(rag_size, Image.NEAREST)
    elif flag == "CROP":
        img = random_crop_area(img) #img.crop((25, 25, width - 25, height - 25))
        return img.resize(rag_size)
    elif flag == "MASK":
        img = ImageOps.grayscale(img)
        return img.resize(rag_size)
    elif flag == "BLUR":
        radius = 2
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        return img.filter(ImageFilter.GaussianBlur(radius)).resize(rag_size)
    elif flag == "ERASE":
        return random_erasing(img).resize(rag_size)
    elif flag == "ROTATE":
        return random_rotate_flip(img).resize(rag_size)
    elif flag == "G-NOISE":
        return add_gaussian_noise(img).resize(rag_size)
    return img

def encode_png(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def encode_rgb(img):
    buffered = BytesIO()
    img.save(buffered, format="RGB")
    return "data:image/rgb;base64," + base64.b64encode(buffered.getvalue()).decode()

def image_hash(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # consistent format
    return hashlib.md5(buf.getvalue()).hexdigest()

def quick_sig(img, n_bytes=1024):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return (img.size, img.mode, hashlib.md5(data[:n_bytes]).hexdigest())
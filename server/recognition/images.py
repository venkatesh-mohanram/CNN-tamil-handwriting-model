"""One canonical PNG-to-model transformation for training and serving."""
from __future__ import annotations
import io
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 1024 * 1024
MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
PREPROCESSING = {"version": 2, "size": 224, "padding": 20, "ink_threshold": 200,
                 "background_percentile": 75, "polarity": "dark-on-light",
                 "mean": [.485,.456,.406], "std": [.229,.224,.225]}

class ImageInputError(ValueError):
    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status = code, status

def decode_png(payload: bytes, max_dimension: int = 1024) -> Image.Image:
    if len(payload) > MAX_BYTES:
        raise ImageInputError("image_too_large", 413, "PNG must be at most 1 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format != "PNG":
                    raise ImageInputError("unsupported_image",415,"Only PNG images are supported.")
                if max(source.size) > max_dimension or min(source.size) < 1 or getattr(source,"n_frames",1) != 1:
                    raise ImageInputError("invalid_dimensions",422,"Use one PNG frame with dimensions at most 1024×1024.")
                rgba = source.convert("RGBA")
                white = Image.new("RGBA",rgba.size,"white")
                return Image.alpha_composite(white,rgba).convert("RGB")
    except ImageInputError:
        raise
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise ImageInputError("invalid_image",422,"Image could not be decoded.") from exc

def normalize_background(image: Image.Image) -> Image.Image:
    """Whiten paper before adding any synthetic borders or geometric augmentation.

    Gray scanned paper can have narrow white scanner strips. A 95th percentile
    estimates those strips rather than the paper, incorrectly marking paper as
    foreground. The upper quartile tolerates those strips while preserving ink.
    This does not infer or invert light-on-dark handwriting.
    """
    gray = np.asarray(ImageOps.grayscale(image),dtype=np.float32)
    background = float(np.percentile(gray,PREPROCESSING["background_percentile"]))
    if background < 1:
        raise ImageInputError("invalid_background",422,"Use dark ink on a light background.")
    gray = np.clip(gray * (255.0/background),0,255).astype(np.uint8)
    # Suppress faint paper texture, but preserve all disconnected ink components.
    gray[gray >= 200] = 255
    return Image.fromarray(gray).convert("RGB")

def normalize_ink(image: Image.Image) -> Image.Image:
    gray = np.asarray(normalize_background(image).convert("L"))
    mask = gray < 200
    if np.count_nonzero(mask) < 8:
        raise ImageInputError("blank_canvas",422,"Draw a character before recognizing.")
    if mask.mean() > .85:
        raise ImageInputError("invalid_ink",422,"Use a single dark character on a light background.")
    ys,xs = np.where(mask)
    crop = Image.fromarray(gray).crop((int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1))
    side = max(crop.size)+40
    square = Image.new("L",(side,side),"white")
    square.paste(crop,((side-crop.width)//2,(side-crop.height)//2))
    return square.resize((224,224),Image.Resampling.LANCZOS).convert("RGB")

def image_tensor(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(normalize_ink(image),dtype=np.float32).transpose(2,0,1)/255.0
    return np.ascontiguousarray(((rgb-MEAN)/STD)[None],dtype=np.float32)

def preprocess_png(payload: bytes) -> np.ndarray:
    return image_tensor(decode_png(payload))

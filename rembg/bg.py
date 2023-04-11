import io
import uuid
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
)
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel, AnyUrl
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion

from .object_pool import ObjectPool
from .session_base import BaseSession
from .session_factory import new_session

from .firebase import upload_blob_from_memory

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))

session_pool = ObjectPool(factory_method=new_session, param="u2net_cloth_seg", max_size=8)


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def alpha_matting_cutout(
    img: PILImage,
    mask: PILImage,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> PILImage:
    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img = np.asarray(img)
    mask = np.asarray(mask)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout


def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary
    return mask


def apply_background_color(img: PILImage, color: Tuple[int, int, int, int]) -> PILImage:
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image
class ClothesImage(BaseModel):
    uri: str
    offset_x: int
    offset_y: int


class SegregatedClothes(BaseModel):
    upper_body: ClothesImage = None
    lower_body: ClothesImage = None
    full_body: ClothesImage = None


def clothes_seg_to_firebase(
    data: bytes,
    alpha_matting: bool = True,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    post_process_mask: bool = False,
) -> List[ClothesImage]:
    img = Image.open(io.BytesIO(data))

    session = session_pool.acquire()
    masks = session.predict(img)
    session_pool.release(session)

    width, height = img.size
    pixel_count = width * height

    cutouts = []
    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                cutout = naive_cutout(img, mask)

        else:
            cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    payload = []
    for sample in cutouts:
        pixels = sample.getdata(3)
        opaque_pixels = 0

        # Loop through all the pixels in the image
        for pixel in pixels:
            if pixel > 5:  # Check the alpha value of the pixel (0 means transparent)
                opaque_pixels += 1

        # Calculate the transparent-to-opaque pixel ratio
        ratio = opaque_pixels / pixel_count
        if ratio < 0.02:
            payload.append(None)
            continue

        bbox = sample.getbbox()
        cropped_img = sample.crop(bbox)

        file_buffer = io.BytesIO()
        cropped_img.save(file_buffer, format="WEBP", exact=True)
        offset_x = ((bbox[2] + bbox[0]) - width) // 2
        offset_y = ((bbox[3] + bbox[1]) - height) // 2
        blob_name = f"{uuid.uuid4()}.webp"
        uri = upload_blob_from_memory(file_buffer.getvalue(), blob_name)
        payload.append(ClothesImage(uri=uri, offset_x=offset_x, offset_y=offset_y))

    return payload


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    session: Optional[BaseSession] = None,
    only_mask: bool = False,
    post_process_mask: bool = False,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
) -> Union[bytes, PILImage, np.ndarray]:
    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    if session is None:
        session = new_session("u2net")

    masks = session.predict(img)
    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask

        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                cutout = naive_cutout(img, mask)

        else:
            cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if bgcolor is not None and not only_mask:
        cutout = apply_background_color(cutout, bgcolor)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()

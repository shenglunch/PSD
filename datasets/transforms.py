import torch
import math
import random

from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None

import numpy as np
import numbers
import types
import collections
import warnings
import skimage.transform
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def rotate(img, angle, interpolation):
    return TF.rotate(img, angle=angle, interpolation=interpolation)

def hflip(img):
    return TF.hflip(img)

class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)

class ToTensor(object):
    def __call__(self, img):
        return T.ToTensor()(img)

class ConvertFloat(object):
    def __call__(self, img):
        return T.ConvertImageDtype(torch.float)(img)

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class NormalizeTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = T.Normalize(self.mean, self.std)(tensor)
        return tensor

class ResizeV1(object):
    def __init__(self, scale, interpolation):
        self.scale = scale
        self.interpolation = interpolation
    
    def __call__(self, img):
        return T.Resize(self.scale, self.interpolation)(img)

class ResizeV2(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """
    def __init__(self, size, order):
        self.size = size
        self.order = order

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be scaled.
        Returns:
            img (numpy.ndarray (C x H x W)): Rescaled image.
        """
        if img.ndim == 3:
            return skimage.transform.resize(img, self.size, order=self.order)
        elif img.ndim == 2:
            return skimage.transform.resize(img, self.size, order=self.order)
        else:
            RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class ResizeV3(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample.size[0], sample.size[1]
        )

        # resize sample
        # sample = cv2.resize(
        #     sample,
        #     (width, height),
        #     interpolation=self.__image_interpolation_method,
        # )

        sample = T.Resize((height, width), self.__image_interpolation_method)(sample)

        return sample

class ResizeV4(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample.shape[1], sample.shape[0]
        )

        # resize sample
        if sample.ndim == 3:
            return skimage.transform.resize(sample, (height, width), order=self.__image_interpolation_method)
        elif sample.ndim == 2:
            return skimage.transform.resize(sample, (height, width), order=self.__image_interpolation_method)
        else:
            RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    sample.ndim))
        return sample

class ColorJitter(object):
    def __init__(self, jitter):
        self.jitter = jitter
    
    def __call__(self, img):
        return T.ColorJitter(brightness=self.jitter, contrast=self.jitter, saturation=self.jitter)(img)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return T.CenterCrop(self.size)(img)

class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for bottom crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))

        # randomized left and right cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-1, j+1)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.

    Args:
        do_flip (boolean): whether or not do horizontal flip.

    """
    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.

        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img).copy()
        else:
            return img



####
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img

def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255)**gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img

class NormalizeNumpyArray(object):
    """Normalize a ``numpy.ndarray`` with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``numpy.ndarray`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image of size (H, W, C) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        # TODO: make efficient
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img

class Rotate(object):
    """Rotates the given ``numpy.ndarray``.

    Args:
        angle (float): The rotation angle in degrees.
    """
    def __init__(self, angle, order):
        self.angle = angle
        self.order = order

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.

        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        # order=0 means nearest-neighbor type interpolation
        # 0: Nearest-neighbor 1: Bi-linear (default) 2: Bi-quadratic 
        # 3: Bi-cubic 4: Bi-quartic 5: Bi-quintic
        return skimage.transform.rotate(img, self.angle, resize=False, order=self.order)

class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """
    def __init__(self, size, order):
        self.size = size
        self.order = order

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be scaled.
        Returns:
            img (numpy.ndarray (C x H x W)): Rescaled image.
        """
        if img.ndim == 3:
            return skimage.transform.resize(img, self.size, order=self.order)
        elif img.ndim == 2:
            return skimage.transform.resize(img, self.size, order=self.order)
        else:
            RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class ResizeShortSide(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """
    def __init__(self, size, order):
        self.size = size
        self.order = order

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be scaled.
        Returns:
            img (numpy.ndarray (C x H x W)): Rescaled image.
        """
        if img.ndim == 3:
            return skimage.transform.resize(img, self.size, order=self.order)
        elif img.ndim == 2:
            return skimage.transform.resize(img, self.size, order=self.order)
        else:
            RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for bottom crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))

        # randomized left and right cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-1, j+1)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class RandomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for bottom crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size

        # randomized left and right cropping
        i = np.random.randint(0, h-th+1)
        j = np.random.randint(0, w-tw+1)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class Crop(object):
    """Crops the given ``numpy.ndarray`` at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, crop):
        self.crop = crop

    @staticmethod
    def get_params(img, crop):
        """Get parameters for ``crop`` for center crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        x_l, x_r, y_b, y_t = crop
        h = img.shape[0]
        w = img.shape[1]
        assert x_l >= 0 and x_l < w
        assert x_r >= 0 and x_r < w
        assert y_b >= 0 and y_b < h
        assert y_t >= 0 and y_t < h
        assert x_l < x_r and y_b < y_t

        return x_l, x_r, y_b, y_t

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        x_l, x_r, y_b, y_t = self.get_params(img, self.crop)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[y_b:y_t, x_l:x_r, :]
        elif img.ndim == 2:
            return img[y_b:y_t, x_l:x_r]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ColorJitterV2(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        transforms = []
        transforms.append(
            Lambda(lambda img: adjust_brightness(img, brightness)))
        transforms.append(Lambda(lambda img: adjust_contrast(img, contrast)))
        transforms.append(
            Lambda(lambda img: adjust_saturation(img, saturation)))
        transforms.append(Lambda(lambda img: adjust_hue(img, hue)))
        np.random.shuffle(transforms)
        self.transform = Compose(transforms)

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Input image.

        Returns:
            img (numpy.ndarray (C x H x W)): Color jittered image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        return np.array(self.transform(pil))


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


class SkinDetector(object):
    """Simple skin segmentation for both with and without lesions"""

    def __init__(self, imageName):
        self.image = cv2.imread(imageName)
        if self.image is None:
            print("Image Not Found")
            exit(1)
        self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        self.binary_mask_image = self.HSV_image

    def find_skin(self):
        """function to process the image and segment the skin using
        the HSV and YCbCr color spaces, followed by the Watershed algorithm"""
        self.color_segmentation()
        image_mask = self.region_based_segmentation()

        return image_mask

    def color_segmentation(self):
        """Apply a threshold to an HSV and YCbCr images,
        the used values were based on current research papers along with some
        empirical tests and visual evaluation"""
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        mask_YCbCr = cv2.inRange(
            self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values
        )
        mask_HSV = cv2.inRange(self.HSV_image, lower_HSV_values, upper_HSV_values)
        self.binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)

    def region_based_segmentation(self):
        """applies Watershed and morphological operations on the thresholded image"""

        image_foreground = cv2.erode(self.binary_mask_image, None, iterations=3)
        dilated_binary_image = cv2.dilate(self.binary_mask_image, None, iterations=3)
        ret, image_background = cv2.threshold(
            dilated_binary_image, 1, 128, cv2.THRESH_BINARY
        )
        image_marker = cv2.add(image_foreground, image_background)
        image_marker32 = np.int32(image_marker)

        cv2.watershed(self.image, image_marker32)
        m = cv2.convertScaleAbs(image_marker32)

        # bitwise of the mask with the input image
        ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        output = cv2.bitwise_and(self.image, self.image, mask=image_mask)

        return image_mask, output


def get_masks_folder(derma_path, output_path, plot=False):
    """return the pixels from the mask
    from a given folder of images"""
    masks = []
    pixels_values = []
    derma_files = glob.glob("{}/**/*.jpeg".format(derma_path), recursive=True)
    for i, d in enumerate(derma_files):
        base = os.path.basename(d)
        detector = SkinDetector(d)
        mask, pixels = detector.find_skin()
        masks.append(mask)
        pixels_values.append(pixels)
        if plot:
            original = plt.imread(d)
            f, axarr = plt.subplots(1, 3)
            axarr[0].set_title("Original")
            axarr[0].imshow(original)
            axarr[1].set_title("Mask")
            axarr[1].imshow(mask, cmap="gray")
            axarr[2].set_title("Selected pixels")
            axarr[2].imshow(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            for a in axarr:
                a.axis("off")
            plt.tight_layout()
            plt.savefig("{}/masks_{}".format(output_path, base))
            plt.show()

    return pixels_values

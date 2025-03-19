import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from bbox import BoundingBox


@dataclass
class DetectionImage:
    """
    A class to handle detection images with associated bounding boxes.

    This class manages image loading and storage of bounding box information

    Attributes:
        bounding_boxes (List[BoundingBox]): List of bounding boxes associated with the image
        filename (Optional[str]): Name of the image file (should always be the same for image and label file)
        image_dir (Optional[Path]): Directory path containing the image
    """

    bounding_boxes: list[BoundingBox]
    filename: Optional[str] = None
    image_dir: Optional[Path] = None

    def get_image(self, image_dir: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Load and return the image from the specified directory.

        Args:
            image_dir (Optional[Union[str, Path]]): Directory containing the image.
                If None, uses the class's image_dir attribute.

        Returns:
            np.ndarray: The loaded image as a NumPy array.

        Raises:
            ValueError: If no image directory is provided either as parameter or class attribute.
            FileNotFoundError: If the image file cannot be found or accessed.
        """
        if not self.filename:
            raise ValueError("Filename is not set")

        # Determine the image directory to use
        working_dir = Path(image_dir) if image_dir else self.image_dir
        if not working_dir:
            raise ValueError("Image directory must be provided either as class attribute or function parameter")

        # Construct full image path
        image_path = os.path.join(working_dir, self.filename)

        # Attempt to load the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {image_path}")

        return image
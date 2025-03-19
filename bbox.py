from __future__ import annotations

class BoundingBox:
    def __init__(self, left: float, top: float, right: float, bottom: float, confidence: float, class_id: int,
                 class_name: str):
        """
        Initialize a BoundingBox object.

        Args:
            left (float): The relative left coordinate of the bounding box.
            top (float): The relative top coordinate of the bounding box.
            right (float): The relative right coordinate of the bounding box.
            bottom (float): The relative bottom coordinate of the bounding box.
            confidence (float): The confidence score associated with the bounding box.
            class_id (int): The class identifier for the object detected within the bounding box. Starts with 0!
            class_name (str): Full class name / label.
        """

        if not all(0.0 <= var <= 1.0 for var in (left, top, right, bottom)):
            raise Exception("BoundingBox initialization failed. "
                            "Bbox l,t,r,b dimensions must be relative and between 0.0 and 1.0.")
        if not 0.0 <= confidence <= 1.0:
            raise Exception("BoundingBox initialization failed. "
                            "Bbox confidence must be between 0.0 and 1.0.")

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        # TODO change name to more generic metric instead of mm
        self.depth_in_mm_GT = None
        self.depth_in_mm_PRED = None

    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        IoU is a measure of the overlap between two bounding boxes. It is calculated as the area of the intersection of the two boxes divided by the area of their union.
        Args:
            box1 (BoundingBox): The first bounding box.
            box2 (BoundingBox): The second bounding box.

        Returns:
            float: The IoU value, which is a float between 0.0 and 1.0.
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1.left, box2.left)
        y_top = max(box1.top, box2.top)
        x_right = min(box1.right, box2.right)
        y_bottom = min(box1.bottom, box2.bottom)

        # Calculate the area of intersection rectangle
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        box1_area = (box1.right - box1.left) * (box1.bottom - box1.top)
        box2_area = (box2.right - box2.left) * (box2.bottom - box2.top)
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou


    def get_dimensions_ltrb_rel(self) -> tuple:
        """
        Get all four dimensions of the bounding box.

        Returns:
            tuple: A tuple containing the left, top, right, and bottom dimensions (all floats).
        """
        return (self.left, self.top, self.right, self.bottom)

    def get_dimensions_ltrb_abs(self, width: int, height: int) -> tuple:
        """
        Get all four dimensions of the bounding box.

        Returns:
            tuple: A tuple containing the left, top, right, and bottom dimensions (all floats).
        """
        left_abs = int(self.left * width)
        top_abs = int(self.top * height)
        right_abs = int(self.right * width)
        bottom_abs = int(self.bottom * height)
        return (left_abs, top_abs, right_abs, bottom_abs)

    def get_dimensions_yolo_rel(self) -> tuple:
        """
        Get all four dimensions of the bounding box. In x,y,w,h yolo format and in relative measurements.

        Returns:
            tuple: A tuple containing the x,y,w,h dimensions (all floats).
        """
        w = self.right - self.left
        h = self.bottom - self.top
        x = self.left + w / 2
        y = self.top + h / 2

        return (x, y, w, h)

    def get_confidence(self) -> float:
        """
        Get the confidence score associated with the bounding box.

        Returns:
            float: The confidence score.
        """
        return self.confidence

    def get_class_id(self) -> int:
        """
        Get the class identifier associated with the bounding box.

        Returns:
            int: The class identifier.
        """
        return self.class_id

    def get_class_name(self) -> int:
        """
        Get the class name / label associated with the bounding box.

        Returns:
            str: The class name / label.
        """
        return self.class_name

    def __str__(self):
        """
        Returns a formatted string representation of the BoundingBox object.

        Returns:
            str: A string representation of the BoundingBox object's attributes.
        """
        attrs = [
            f"left: {self.left:.2f}",
            f"top: {self.top:.2f}",
            f"right: {self.right:.2f}",
            f"bottom: {self.bottom:.2f}",
            f"confidence: {self.confidence:.2f}",
            f"class_id: {self.class_id}",
            f"class_name: {self.class_name}"
        ]
        if self.depth_in_mm_GT is not None:
            attrs.append(f"depth_in_mm_GT: {self.depth_in_mm_GT}")
        if self.depth_in_mm_PRED is not None:
            attrs.append(f"depth_in_mm_PRED: {self.depth_in_mm_PRED}")
        return "BoundingBox(" + ", ".join(attrs) + ")"
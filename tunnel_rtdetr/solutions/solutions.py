
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import cv2
import numpy as np

from tunnel_rtdetr import RTDETR
from tunnel_rtdetr.solutions.config import SolutionConfig
from tunnel_rtdetr.utils import LOGGER, ops
from tunnel_rtdetr.utils.checks import check_imshow, check_requirements
from tunnel_rtdetr.utils.plotting import Annotator


class BaseSolution:
    """BaseSolution. RT-DETR detection workflow."""

    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        """init. RT-DETR detection workflow."""
        self.CFG = vars(SolutionConfig().update(**kwargs))
        self.LOGGER = LOGGER  # Store logger object to be used in multiple solution classes

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # Initialize annotator
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1  # Only for logging

        self.LOGGER.info(f"Tunnel RT-DETR Solutions: ✅ {self.CFG}")
        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = self.CFG["line_width"]

        # Load Model and store additional information (classes, show_conf, show_label)
        if self.CFG["model"] is None:
            self.CFG["model"] = "rtdetr-l.pt"
        self.model = RTDETR(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.show_conf = self.CFG["show_conf"]
        self.show_labels = self.CFG["show_labels"]
        self.device = self.CFG["device"]

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in {"iou", "conf", "device", "max_det", "half", "tracker"}
        }  # verbose must be passed to track method; setting it False in YOLO still logs the track information.

        if is_cli and self.CFG["source"] is None:
            raise ValueError("An explicit video source is required for counting.")

        # Initialize environment and region setup
        self.env_check = check_imshow(warn=True) if self.CFG["show"] else False
        self.track_history = defaultdict(list)

        self.profilers = (
            ops.Profile(device=self.device),  # track
            ops.Profile(device=self.device),  # solution
        )

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        """Generate a formatted label for a bounding box.

        This method constructs a label string for a bounding box using the class index and confidence score. Optionally
        includes the track ID if provided. The label format adapts based on the display settings defined in
        `self.show_conf` and `self.show_labels`.

        Args:
            cls (int): The class index of the detected object.
            conf (float): The confidence score of the detection.
            track_id (int, optional): The unique identifier for the tracked object.

        Returns:
            (str | None): The formatted label string if `self.show_labels` is True; otherwise, None.
        """
        name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None

    def extract_tracks(self, im0: np.ndarray) -> None:
        """Apply object tracking and extract tracks from an input image or frame.

        Args:
            im0 (np.ndarray): The input image or frame.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        self.track_data = self.tracks.boxes

        if self.track_data and self.track_data.is_track:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.LOGGER.warning("No tracks found.")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    def store_tracking_history(self, track_id: int, box) -> None:
        """Store the tracking history of an object.

        This method updates the tracking history for a given object by appending the center point of its bounding box to
        the track line. It maintains a maximum of 30 points in the tracking history.

        Args:
            track_id (int): The unique identifier for the tracked object.
            box (list[float]): The bounding box coordinates of the object in the format [x1, y1, x2, y2].

        Examples:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]
        coordinates = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
        center = coordinates.reshape(-1, 2).mean(axis=0)
        self.track_line.append(tuple(float(value) for value in center))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self) -> None:
        """Initialize the counting region and line segment based on configuration settings."""
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # region or line

    def display_output(self, plot_im: np.ndarray) -> None:
        """Display the results of the processing, which could involve showing frames, printing counts, or saving
        results.

        This method is responsible for visualizing the output of the object detection and tracking process. It displays
        the processed frame with annotations, and allows for user interaction to close the display.

        Args:
            plot_im (np.ndarray): The image or frame that has been processed and annotated.

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        Notes:
            - This method will only display output if the 'show' configuration is set to True and the environment
              supports image display.
            - The display can be closed by pressing the 'q' key.
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Tunnel RT-DETR Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # Closes current frame window
                return

    def process(self, *args: Any, **kwargs: Any):
        """Process method should be implemented by each Solution subclass."""

    def __call__(self, *args: Any, **kwargs: Any):
        """Allow instances to be called like a function with flexible arguments."""
        with self.profilers[1]:
            result = self.process(*args, **kwargs)  # Call the subclass-specific process method
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # solution time = process - track
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)  # Only for logging.
            LOGGER.info(
                f"{self.frame_no}: {result.plot_im.shape[0]}x{result.plot_im.shape[1]} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape "
                f"(1, {getattr(self.model, 'ch', 3)}, {result.plot_im.shape[0]}, {result.plot_im.shape[1]})\n"
            )
        return result


class SolutionAnnotator(Annotator):
    """SolutionAnnotator. RT-DETR detection workflow."""

    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the SolutionAnnotator class with an image for annotation.

        Args:
            im (np.ndarray): The image to be annotated.
            line_width (int, optional): Line thickness for drawing on the image.
            font_size (int, optional): Font size for text annotations.
            font (str): Path to the font file.
            pil (bool): Indicates whether to use PIL for rendering text.
            example (str): Example text used to detect non-ASCII labels for PIL rendering.
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(
        self,
        reg_pts: list[tuple[int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5,
    ):
        """Draw a region or line on the image.

        Args:
            reg_pts (list[tuple[int, int]], optional): Region points (for line 2 points, for region 4+ points).
            color (tuple[int, int, int]): BGR color value for the region (OpenCV format).
            thickness (int): Line thickness for drawing the region.
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle


    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
        position: str = "top-right",
    ):
        """Display overall statistics for Solutions (e.g., parking management and object counting).

        Args:
            im0 (np.ndarray): Inference image.
            text (dict[str, Any]): Labels dictionary.
            txt_color (tuple[int, int, int]): Text color (BGR, OpenCV format).
            bg_color (tuple[int, int, int]): Background color (BGR, OpenCV format).
            margin (int): Gap between text and rectangle for better display.
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            if position == "top-left":
                text_x = margin * 2 + horizontal_gap
            else:
                text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2













class SolutionResults:
    """SolutionResults. RT-DETR detection workflow."""

    def __init__(self, plot_im=None, in_count=0, out_count=0, classwise_count=None, total_tracks=0):
        self.plot_im = plot_im
        self.in_count, self.out_count = in_count, out_count
        self.classwise_count = classwise_count or {}
        self.total_tracks = total_tracks
        self.speed = {}

    def __str__(self) -> str:
        """Return a formatted string representation of the SolutionResults object.

        Returns:
            (str): A string representation listing non-null attributes.
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # Exclude `plot_im` explicitly
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())

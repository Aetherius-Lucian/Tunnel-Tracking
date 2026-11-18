"""results. RT-DETR detection workflow."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from tunnel_rtdetr.utils import DataExportMixin, SimpleClass, ops
from tunnel_rtdetr.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    """BaseTensor. RT-DETR detection workflow."""

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """init. RT-DETR detection workflow."""
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data tensor.

        Returns:
            (tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor stored in CPU memory.

        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of this object with its data converted to a NumPy array.

        Returns:
            (BaseTensor): A new instance with `data` as a NumPy array.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Move the tensor to GPU memory.

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory.

        Examples:
            >>> import torch
            >>> from tunnel_rtdetr.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """Return the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | list[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass, DataExportMixin):
    """Results. RT-DETR detection workflow."""

    def __init__(self, orig_img, path, names, boxes=None, speed=None):
        self.orig_img, self.orig_shape = orig_img, orig_img.shape[:2]
        self.path, self.names = path, names
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.speed = speed or {"preprocess": None, "inference": None, "postprocess": None}
        self.save_dir = None
        self._keys = ("boxes",)

    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        return self._apply("__getitem__", idx)

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0

    def update(self, boxes=None):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a function to all non-empty attributes and return a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args (Any): Variable length argument list to pass to the function.
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """cpu. RT-DETR detection workflow."""
        return self._apply("cpu")

    def numpy(self):
        """Convert all tensors in the Results object to numpy arrays.

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.
        """
        return self._apply("numpy")

    def cuda(self):
        """Move all tensors in the Results object to GPU memory.

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # Move first result to GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # Move each result to GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Move all tensors in the Results object to the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
            >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(self, conf=True, line_width=None, font_size=None, font="Arial.ttf", pil=False,
             img=None, labels=True, boxes=True, show=False, save=False, filename=None,
             color_mode="class", txt_color=(255, 255, 255)):
        """Draw RT-DETR boxes, class names, confidence and optional tracking IDs."""
        if color_mode not in {"class", "instance"}:
            raise ValueError("color_mode must be class or instance")
        annotator = Annotator(deepcopy(self.orig_img if img is None else img),
                              line_width, font_size, font, pil, example=self.names)
        if boxes and self.boxes is not None:
            for index, box in enumerate(reversed(self.boxes)):
                cls = int(box.cls)
                track_id = int(box.id.item()) if box.is_track else None
                name = (f"id:{track_id} " if track_id is not None else "") + self.names[cls]
                label = (f"{name} {float(box.conf):.2f}" if conf else name) if labels else None
                color_id = cls if color_mode == "class" else (track_id if track_id is not None else index)
                annotator.box_label(box.xyxy.squeeze(), label, color=colors(color_id, True), txt_color=txt_color)
        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or Path(self.path).name)
        return annotator.result(pil=pil)

    def show(self, *args, **kwargs):
        """Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            >>>     result.show()  # Display all results
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename: str | None = None, *args, **kwargs) -> str:
        """Save annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | Path | None): The filename to save the annotated image. If None, a default filename is
                generated based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Returns:
            (str): The filename where the image was saved.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save("annotated_image.jpg")
            >>> # Or with custom plot arguments
            >>> for result in results:
            >>>     result.save("annotated_image.jpg", conf=False, line_width=2)
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        if not len(self):
            return "(no detections), "
        counts = self.boxes.cls.int().bincount() if isinstance(self.boxes.data, torch.Tensor) else np.bincount(self.boxes.cls.astype(int))
        return "".join(f"{int(count)} {self.names[i]}{'s' if count > 1 else ''}, " for i, count in enumerate(counts) if count)

    def save_txt(self, txt_file, save_conf=False):
        lines = []
        if self.boxes is not None:
            for box in self.boxes:
                row = (int(box.cls), *box.xywhn.reshape(-1).tolist())
                if save_conf:
                    row += (float(box.conf),)
                if box.is_track:
                    row += (int(box.id.item()),)
                lines.append(" ".join(f"{x:g}" for x in row))
        if lines:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        return str(txt_file)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        if self.boxes is not None:
            for box in self.boxes:
                save_one_box(box.xyxy, self.orig_img.copy(),
                             file=Path(save_dir) / self.names[int(box.cls)] / Path(file_name).with_suffix(".jpg"),
                             BGR=True)

    def summary(self, normalize=False, decimals=5):
        output = []
        height, width = self.orig_shape if normalize else (1, 1)
        if self.boxes is None:
            return output
        for box in self.boxes:
            cls = int(box.cls)
            coords = box.xyxy.reshape(-1).tolist()
            entry = {"name": self.names[cls], "class": cls, "confidence": round(float(box.conf), decimals),
                     "box": {key: round(value / scale, decimals) for key, value, scale in
                             zip(("x1", "y1", "x2", "y2"), coords, (width, height, width, height))}}
            if box.is_track:
                entry["track_id"] = int(box.id.item())
            output.append(entry)
        return output


class Boxes(BaseTensor):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence
        scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates,
        including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Return bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box coordinates in
                [x1, y1, x2, y2] format, where n is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence scores for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence scores for each detection, with
                shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """Return the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The
                shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """Return the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled, otherwise None.
                Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> torch.Tensor | np.ndarray:
        """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center
                are the coordinates of the center point of the bounding box, width, height are the dimensions of the
                bounding box and the shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[125.0000,  75.0000,  50.0000,  50.0000],
                    [250.0000, 200.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format, normalized to the
        range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is the number of
                boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
        width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N, 4), where N is the number of boxes.
                Each row contains [x_center, y_center, width, height] values normalized to [0, 1] based on the original
                image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh









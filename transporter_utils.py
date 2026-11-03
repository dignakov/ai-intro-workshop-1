from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 320
CROP_SIZE = 160
BATCH_SIZE = 4


@dataclass
class PickPlaceSample:
    image: np.ndarray
    pick: tuple[float, float]
    place: tuple[float, float]
    pick_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    pick: tuple[float, float]
    place: tuple[float, float]
    pick_heatmap: np.ndarray
    place_heatmap: np.ndarray


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_box(
    image: np.ndarray,
    x: float,
    y: float,
    size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = image.copy()
    half = size // 2
    x, y = int(round(x)), int(round(y))
    result[y - half : y + half, x - half : x + half] = color
    return result


def _toy_sample(rng: np.random.Generator, image_size: int) -> PickPlaceSample:
    blue_color = (78, 121, 167)
    red_color = (225, 87, 89)

    for _ in range(1000):
        blue_size, red_size = (rng.integers(14, 33, 2) * 2).tolist()
        blue_radius = blue_size / 2
        red_radius = red_size / 2
        blue = (
            rng.uniform(blue_radius, image_size - blue_radius),
            rng.uniform(blue_radius, image_size - blue_radius),
        )
        place = (
            blue[0] - (blue_size + red_size) / 2,
            blue[1] - (blue_size - red_size) / 2,
        )
        if not (
            red_radius <= place[0] < image_size - red_radius
            and red_radius <= place[1] < image_size - red_radius
        ):
            continue

        pick = (
            rng.uniform(red_radius, image_size - red_radius),
            rng.uniform(red_radius, image_size - red_radius),
        )
        separated = (
            abs(pick[0] - blue[0]) > (blue_size + red_size) / 2
            or abs(pick[1] - blue[1]) > (blue_size + red_size) / 2
        )
        if separated and math.dist(pick, place) > red_size:
            break
    else:
        raise RuntimeError("Could not generate a valid toy sample")

    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    image = draw_box(image, *blue, blue_size, blue_color)
    image = draw_box(image, *pick, red_size, red_color)
    return PickPlaceSample(
        image=image,
        pick=pick,
        place=place,
        pick_radius=red_size / math.sqrt(2),
        metadata={
            "kind": "toy",
            "blue": blue,
            "blue_size": blue_size,
            "red_size": red_size,
        },
    )


def render_toy_goal(sample: PickPlaceSample) -> np.ndarray:
    if sample.metadata.get("kind") != "toy":
        raise ValueError("Goal rendering is available for toy samples")
    image = np.full_like(sample.image, 255)
    image = draw_box(
        image,
        *sample.metadata["blue"],
        sample.metadata["blue_size"],
        (78, 121, 167),
    )
    return draw_box(
        image,
        *sample.place,
        sample.metadata["red_size"],
        (225, 87, 89),
    )


def _transform_points(
    points: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    return homogeneous @ matrix.T


def _augment(
    sample: PickPlaceSample,
    rng: np.random.Generator,
    allow_flips: bool,
) -> PickPlaceSample:
    image = sample.image
    height, width = image.shape[:2]
    points = np.array([sample.pick, sample.place], dtype=np.float32)
    pick_radius = sample.pick_radius

    angle = np.deg2rad(rng.uniform(-12, 12))
    scale = rng.uniform(0.92, 1.08)
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    center_x, center_y = width / 2, height / 2
    matrix = np.array(
        [
            [alpha, beta, (1 - alpha) * center_x - beta * center_y],
            [-beta, alpha, beta * center_x + (1 - alpha) * center_y],
        ],
        dtype=np.float32,
    )
    matrix[0, 2] += rng.uniform(-0.04, 0.04) * width
    matrix[1, 2] += rng.uniform(-0.04, 0.04) * height
    transformed = _transform_points(points, matrix)
    inside = np.all((transformed >= 2) & (transformed < [width - 2, height - 2]))
    if inside:
        inverse = np.linalg.inv(np.vstack((matrix, [0, 0, 1])))
        fill = tuple(np.median(image.reshape(-1, 3), axis=0).astype(int))
        image = np.asarray(
            Image.fromarray(image).transform(
                (width, height),
                Image.Transform.AFFINE,
                tuple(inverse[:2].flatten()),
                resample=Image.Resampling.BILINEAR,
                fillcolor=fill,
            )
        )
        points = transformed
        pick_radius *= scale

    if allow_flips and rng.random() < 0.5:
        image = np.fliplr(image)
        points[:, 0] = width - 1 - points[:, 0]
    if allow_flips and rng.random() < 0.5:
        image = np.flipud(image)
        points[:, 1] = height - 1 - points[:, 1]

    image_float = image.astype(np.float32)
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-12, 12)
    image_float = (image_float - 127.5) * contrast + 127.5 + brightness
    image = np.clip(image_float, 0, 255).astype(np.uint8)

    return PickPlaceSample(
        image=np.ascontiguousarray(image),
        pick=tuple(points[0]),
        place=tuple(points[1]),
        pick_radius=pick_radius,
        metadata=dict(sample.metadata),
    )


def canonicalize(
    sample: PickPlaceSample,
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    allow_flips: bool = False,
) -> PickPlaceSample:
    height, width = sample.image.shape[:2]
    scale = min(image_size / width, image_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(sample.image).resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
    )
    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2
    image = np.pad(
        resized,
        (
            (pad_y, image_size - new_height - pad_y),
            (pad_x, image_size - new_width - pad_x),
            (0, 0),
        ),
        mode="reflect",
    )
    metadata = dict(sample.metadata)
    metadata["canonical_transform"] = {
        "original_size": (width, height),
        "scale": scale,
        "pad": (pad_x, pad_y),
        "image_size": image_size,
    }
    result = PickPlaceSample(
        image=image,
        pick=(sample.pick[0] * scale + pad_x, sample.pick[1] * scale + pad_y),
        place=(sample.place[0] * scale + pad_x, sample.place[1] * scale + pad_y),
        pick_radius=sample.pick_radius * scale,
        metadata=metadata,
    )
    if augment:
        result = _augment(result, rng or np.random.default_rng(), allow_flips)
    return result


def to_original_coordinates(
    sample: PickPlaceSample, point: Sequence[float]
) -> tuple[float, float]:
    transform = sample.metadata["canonical_transform"]
    scale = transform["scale"]
    pad_x, pad_y = transform["pad"]
    return (
        (float(point[0]) - pad_x) / scale,
        (float(point[1]) - pad_y) / scale,
    )


class _BaseDataset(Dataset):
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
        allow_flips: bool = False,
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.allow_flips = allow_flips
        self.rng = np.random.default_rng(seed)

    def _get_sample(self, index: int) -> PickPlaceSample:
        raise NotImplementedError

    def __getitem__(self, index: int) -> PickPlaceSample:
        sample = self._get_sample(index)
        return canonicalize(
            sample,
            image_size=self.image_size,
            augment=self.augment,
            rng=self.rng,
            allow_flips=self.allow_flips,
        )


class ToyDataset(_BaseDataset):
    def __init__(
        self,
        n: int = 100,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(image_size, augment, seed, allow_flips=False)
        generator = np.random.default_rng(seed)
        self.samples = [_toy_sample(generator, image_size) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample(self, index: int) -> PickPlaceSample:
        return self.samples[index]


class RealDataset(_BaseDataset):
    def __init__(
        self,
        root_dir: str | Path,
        json_file: str = "labels.json",
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
        allow_flips: bool = True,
    ) -> None:
        super().__init__(image_size, augment, seed, allow_flips)
        self.root = Path(root_dir)
        with (self.root / json_file).open() as file:
            records = json.load(file)
        self.records = []
        for sample_id, record in records.items():
            image_name = record.get("image", record.get("image_start"))
            place = record.get("stop", record.get("end"))
            if image_name is None or place is None:
                raise ValueError(f"Missing image or place label in {sample_id}")
            self.records.append(
                {
                    "sample_id": sample_id,
                    "image": image_name,
                    "image_size": tuple(record["image_size"]),
                    "pick": tuple(record["start"]),
                    "place": tuple(place),
                    "pick_radius": float(record["start_radius"]),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def _get_sample(self, index: int) -> PickPlaceSample:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as source:
            source.load()
            if tuple(record["image_size"]) == source.size:
                image = source.convert("RGB")
            else:
                image = ImageOps.exif_transpose(source).convert("RGB")
            if tuple(record["image_size"]) != image.size:
                raise ValueError(
                    f"Label size {record['image_size']} does not match "
                    f"{record['image']} with size {image.size}"
                )
            image_np = np.asarray(image).copy()
        return PickPlaceSample(
            image=image_np,
            pick=record["pick"],
            place=record["place"],
            pick_radius=record["pick_radius"],
            metadata={
                "kind": "real",
                "sample_id": record["sample_id"],
                "image_path": str(self.root / record["image"]),
            },
        )


def suggest_crop_size(
    dataset: Dataset,
    augmentation_scale: float = 1.08,
    safety_margin: float = 1.20,
    multiple: int = 16,
) -> int:
    """Suggest a fixed crop that contains the largest labeled object."""
    if len(dataset) == 0:
        raise ValueError("Cannot suggest a crop size for an empty dataset")
    if min(augmentation_scale, safety_margin, multiple) <= 0:
        raise ValueError("Crop-size factors must be positive")

    previous_augment = getattr(dataset, "augment", None)
    if previous_augment is not None:
        dataset.augment = False
    try:
        max_radius = max(dataset[index].pick_radius for index in range(len(dataset)))
    finally:
        if previous_augment is not None:
            dataset.augment = previous_augment

    required = 2 * max_radius * augmentation_scale * safety_margin
    return math.ceil(required / multiple) * multiple


def crop_square(
    image: np.ndarray, center: Sequence[float], crop_size: int = CROP_SIZE
) -> np.ndarray:
    x, y = int(round(center[0])), int(round(center[1]))
    half = crop_size // 2
    x0, y0 = x - half, y - half
    x1, y1 = x0 + crop_size, y0 + crop_size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image.shape[1])
    pad_bottom = max(0, y1 - image.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return image[y0:y1, x0:x1]


def normalize_image(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def gaussian_heatmap(
    shape: tuple[int, int], center: Sequence[float], sigma: float = 2.0
) -> np.ndarray:
    height, width = shape
    x = np.arange(width, dtype=np.float32) - float(center[0])
    y = np.arange(height, dtype=np.float32) - float(center[1])
    heatmap = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    heatmap /= heatmap.sum() + 1e-12
    return heatmap.astype(np.float32)


def make_batch(
    samples: Sequence[PickPlaceSample],
    crop_size: int = CROP_SIZE,
    sigma: float = 2.0,
) -> dict[str, torch.Tensor]:
    images = torch.stack([normalize_image(sample.image) for sample in samples])
    queries = torch.stack(
        [
            normalize_image(crop_square(sample.image, sample.pick, crop_size))
            for sample in samples
        ]
    )
    pick_targets = torch.from_numpy(
        np.stack(
            [
                gaussian_heatmap(sample.image.shape[:2], sample.pick, sigma)
                for sample in samples
            ]
        )
    )
    place_targets = torch.from_numpy(
        np.stack(
            [
                gaussian_heatmap(sample.image.shape[:2], sample.place, sigma)
                for sample in samples
            ]
        )
    )
    return {
        "image": images,
        "query": queries,
        "pick_target": pick_targets,
        "place_target": place_targets,
        "pick_xy": torch.tensor([sample.pick for sample in samples], dtype=torch.float32),
        "place_xy": torch.tensor([sample.place for sample in samples], dtype=torch.float32),
    }


def spatial_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0]
    log_probabilities = F.log_softmax(logits.flatten(1), dim=1)
    return -(targets.flatten(1) * log_probabilities).sum(dim=1).mean()


def heatmap_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    flat_indices = heatmap.flatten(1).argmax(dim=1)
    width = heatmap.shape[-1]
    return torch.stack((flat_indices % width, flat_indices // width), dim=1).float()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(4, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureUNet(nn.Module):
    def __init__(
        self, in_channels: int = 3, output_channels: int = 16, base_channels: int = 12
    ) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 4)
        self.dec3 = ConvBlock(b * 8, b * 2)
        self.dec2 = ConvBlock(b * 4, b)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, output_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(
            torch.cat((F.interpolate(bottleneck, size=e3.shape[-2:], mode="bilinear"), e3), 1)
        )
        d2 = self.dec2(
            torch.cat((F.interpolate(d3, size=e2.shape[-2:], mode="bilinear"), e2), 1)
        )
        d1 = self.dec1(
            torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear"), e1), 1)
        )
        return self.head(d1)


class AttentionNet(nn.Module):
    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        self.backbone = FeatureUNet(3, 1, base_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(1)


class TransporterNet(nn.Module):
    def __init__(self, feature_dim: int = 12, base_channels: int = 12) -> None:
        super().__init__()
        self.query_encoder = FeatureUNet(3, feature_dim, base_channels)
        self.key_encoder = FeatureUNet(3, feature_dim, base_channels)

    def forward(self, query: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        query_features = self.query_encoder(query)
        key_features = self.key_encoder(image)
        crop_height, crop_width = query_features.shape[-2:]
        pad_y, pad_x = crop_height // 2, crop_width // 2
        key_features = F.pad(key_features, (pad_x, pad_x, pad_y, pad_y))
        outputs = []
        for index in range(query.shape[0]):
            kernel = F.pad(query_features[index], (0, 1, 0, 1))
            correlation = F.conv2d(
                key_features[index : index + 1],
                kernel.unsqueeze(1),
                groups=kernel.shape[0],
            )
            outputs.append(correlation.sum(dim=1)[0])
        return torch.stack(outputs) / (crop_height * crop_width)


def _loader(
    dataset: Dataset, batch_size: int, shuffle: bool, crop_size: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda samples: make_batch(samples, crop_size),
    )


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_attention(
    model: AttentionNet,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    crop_size: int = CROP_SIZE,
) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, False, crop_size):
            batch = _move_batch(batch, device)
            predicted = heatmap_argmax(model(batch["image"]))
            errors.append(torch.linalg.vector_norm(predicted - batch["pick_xy"], dim=1))
    model.train()
    return torch.cat(errors).mean().item()


def evaluate_transporter(
    model: TransporterNet,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    crop_size: int = CROP_SIZE,
) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, False, crop_size):
            batch = _move_batch(batch, device)
            predicted = heatmap_argmax(model(batch["query"], batch["image"]))
            errors.append(torch.linalg.vector_norm(predicted - batch["place_xy"], dim=1))
    model.train()
    return torch.cat(errors).mean().item()


def train_attention(
    model: AttentionNet,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    device: torch.device | None = None,
    epochs: int = 10,
    batch_size: int = BATCH_SIZE,
    crop_size: int = CROP_SIZE,
    learning_rate: float = 1e-3,
    report_every: int = 1,
) -> list[dict[str, float]]:
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _loader(train_dataset, batch_size, True, crop_size):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = spatial_cross_entropy(model(batch["image"]), batch["pick_target"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % report_every == 0 or epoch == epochs:
            result = {"epoch": epoch, "loss": float(np.mean(losses))}
            message = f"attention epoch {epoch:3d} | loss {result['loss']:.4f}"
            if val_dataset is not None:
                result["val_error"] = evaluate_attention(
                    model, val_dataset, device, batch_size, crop_size
                )
                message += f" | val pick error {result['val_error']:.2f}px"
            history.append(result)
            print(message)
    return history


def train_transporter(
    model: TransporterNet,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    device: torch.device | None = None,
    epochs: int = 10,
    batch_size: int = BATCH_SIZE,
    crop_size: int = CROP_SIZE,
    learning_rate: float = 1e-3,
    report_every: int = 1,
) -> list[dict[str, float]]:
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _loader(train_dataset, batch_size, True, crop_size):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = spatial_cross_entropy(
                model(batch["query"], batch["image"]), batch["place_target"]
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % report_every == 0 or epoch == epochs:
            result = {"epoch": epoch, "loss": float(np.mean(losses))}
            message = f"transport epoch {epoch:3d} | loss {result['loss']:.4f}"
            if val_dataset is not None:
                result["val_error"] = evaluate_transporter(
                    model, val_dataset, device, batch_size, crop_size
                )
                message += f" | val place error {result['val_error']:.2f}px"
            history.append(result)
            print(message)
    return history


@torch.no_grad()
def predict(
    attention: AttentionNet,
    transporter: TransporterNet,
    sample: PickPlaceSample,
    device: torch.device | None = None,
    crop_size: int = CROP_SIZE,
) -> Prediction:
    device = device or get_device()
    attention.eval()
    transporter.eval()
    image = normalize_image(sample.image).unsqueeze(0).to(device)
    pick_logits = attention(image)
    pick = heatmap_argmax(pick_logits)[0].cpu().numpy()
    query_image = crop_square(sample.image, pick, crop_size)
    query = normalize_image(query_image).unsqueeze(0).to(device)
    place_logits = transporter(query, image)
    place = heatmap_argmax(place_logits)[0].cpu().numpy()
    return Prediction(
        pick=tuple(pick),
        place=tuple(place),
        pick_heatmap=pick_logits[0].cpu().numpy(),
        place_heatmap=place_logits[0].cpu().numpy(),
    )


def evaluate_pipeline(
    attention: AttentionNet,
    transporter: TransporterNet,
    dataset: Dataset,
    device: torch.device | None = None,
    crop_size: int = CROP_SIZE,
) -> dict[str, float]:
    device = device or get_device()
    pick_errors, place_errors = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = predict(attention, transporter, sample, device, crop_size)
        pick_errors.append(math.dist(prediction.pick, sample.pick))
        place_errors.append(math.dist(prediction.place, sample.place))
    return {
        "pick_error": float(np.mean(pick_errors)),
        "place_error": float(np.mean(place_errors)),
    }


def save_checkpoint(
    attention: AttentionNet,
    transporter: TransporterNet,
    path: str | Path,
    image_size: int = IMAGE_SIZE,
    crop_size: int = CROP_SIZE,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "attention": attention.state_dict(),
            "transporter": transporter.state_dict(),
            "image_size": image_size,
            "crop_size": crop_size,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[AttentionNet, TransporterNet, dict[str, int]]:
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    attention = AttentionNet().to(device)
    transporter = TransporterNet().to(device)
    attention.load_state_dict(checkpoint["attention"])
    transporter.load_state_dict(checkpoint["transporter"])
    attention.eval()
    transporter.eval()
    config = {
        "image_size": checkpoint["image_size"],
        "crop_size": checkpoint["crop_size"],
    }
    return attention, transporter, config


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

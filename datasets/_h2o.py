##
##
##

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import torch
from PIL import Image
from typing_extensions import override

from ._dataset import Dataset
from ._types import Annotation, Sample


class H2ODataset(Dataset):
    """The Human-Human-Object (H2O) Interaction Detection dataset."""

    def __init__(self, root: Union[Path, str], split: Literal["train", "test"]) -> None:
        super().__init__()

        self._root = Path(root)
        self._split = split

        self._samples = self._get_samples()
        self._entity_classes = self._get_entity_classes()
        self._interaction_classes = self._get_interaction_classes()

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    @override
    def human_class_index(self) -> int:
        return self._entity_classes["person"]

    @override
    def entity_classes(self) -> Tuple[str, ...]:
        return tuple(self._entity_classes.keys())

    @override
    def interaction_classes(self) -> Tuple[str, ...]:
        return tuple(self._interaction_classes.keys())

    @override
    def valid_interaction_tuples(self) -> Optional[List[List[int]]]:
        file = self._root / "interaction_tuples.json"
        if not file.exists():
            return None

        with file.open() as f:
            data = json.load(f)

        interaction_tuples = [[] for _ in range(len(self._entity_classes))]
        for couple in data:
            entity_id = self._entity_classes[couple["object"]]
            interaction_id = self._interaction_classes[couple["interaction"]]

            interaction_tuples[entity_id].append(interaction_id)

        return interaction_tuples

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @override
    def __getitem__(self, idx: int) -> Tuple[Sample, Annotation]:
        sample = self._samples[idx]

        image_path = self._root / "images" / self._split / f"{sample['id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        entity_coordinates = []
        entity_labels = []

        for entity in sample["entities"]:
            entity_coordinates.append(entity["bbox"])
            entity_labels.append(self._entity_classes[entity["label"]])

        entity_coordinates = torch.as_tensor(entity_coordinates, dtype=torch.float32)
        entity_labels = torch.as_tensor(entity_labels, dtype=torch.int64)

        # coordinates are normalized to [0, 1], but we need them in [0, W] and [0, H]
        entity_coordinates[:, [0, 2]] *= image.width
        entity_coordinates[:, [1, 3]] *= image.height

        inter_indices = []
        inter_classes = []

        custom_inter_indices = torch.empty(
            (len(sample["interactions"]), 2), dtype=torch.long
        )
        custom_inter_labels = torch.zeros(
            (len(sample["interactions"]), len(self._interaction_classes)),
            dtype=torch.float,
        )

        for idx, interaction in enumerate(sample["interactions"]):
            subject_idx = interaction["subject_idx"]
            object_idx = interaction["object_idx"]
            custom_inter_indices[idx, 0] = subject_idx
            custom_inter_indices[idx, 1] = object_idx

            for inter_label in interaction["labels"]:
                inter_indices.append((subject_idx, object_idx))  # type: ignore
                inter_classes.append(self._interaction_classes[inter_label])
                custom_inter_labels[idx, self._interaction_classes[inter_label]] = 1.0

        inter_indices = torch.as_tensor(inter_indices, dtype=torch.long)  # (N, 2)
        inter_classes = torch.as_tensor(inter_classes, dtype=torch.long)  # (N,)

        if len(inter_indices) > 0:
            boxes_h = entity_coordinates[inter_indices[:, 0]]
            boxes_o = entity_coordinates[inter_indices[:, 1]]
            objects = entity_labels[inter_indices[:, 1]]
        else:
            boxes_h = torch.empty((0, 4), dtype=torch.float32)
            boxes_o = torch.empty((0, 4), dtype=torch.float32)
            objects = torch.empty((0,), dtype=torch.long)

        target: Annotation = {
            "boxes_h": boxes_h,
            "boxes_o": boxes_o,
            "object": objects,
            "labels": inter_classes,
            "custom_entity_boxes": entity_coordinates,
            "custom_entity_labels": entity_labels,
            "custom_interaction_indices": custom_inter_indices,
            "custom_interaction_labels": custom_inter_labels,
        }

        return image, target

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    def _get_samples(self) -> List[_SampleData]:
        with open(self._root / f"{self._split}.json") as file:
            data = json.load(file)

        # In H2O all samples have at least one entity
        return data

    def _get_entity_classes(self) -> Dict[str, int]:
        with open(self._root / "entity_classes.json") as file:
            classes: list[str] = json.load(file)

        return {name: i for i, name in enumerate(classes)}

    def _get_interaction_classes(self) -> Dict[str, int]:
        with open(self._root / "interaction_classes.json") as file:
            classes: list[str] = json.load(file)

        return {name: i for i, name in enumerate(classes)}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


class _SampleData(TypedDict):
    id: str
    entities: List[_EntityData]
    interactions: List[_InteractionData]


class _EntityData(TypedDict):
    label: str
    bbox: List[float]


class _InteractionData(TypedDict):
    subject_idx: int
    object_idx: int
    instruments_idx: List[Optional[int]]
    labels: List[str]

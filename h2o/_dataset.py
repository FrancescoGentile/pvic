##
##
##

import json
from typing import Literal, TypedDict, List, Dict, Tuple, Union
from pathlib import Path

import torch
from PIL import Image

# -------------------------------------------------------------------------- #
# Types
# -------------------------------------------------------------------------- #


Entity = TypedDict("Entity", {"bbox": List[float], "class": str})


class Interaction(TypedDict):
    subject_idx: int
    object_idx: int
    instruments_idx: List[Union[int, None]]
    classes: List[str]


class FileSample(TypedDict):
    id: str
    entities: List[Entity]
    interactions: List[Interaction]


# -------------------------------------------------------------------------- #
# H2ODataset
# -------------------------------------------------------------------------- #


class H2ODataset:
    def __init__(self, path: Union[Path, str], split: Literal["train", "test"]) -> None:
        """Initializes a new H2O dataset.

        Args:
            path: The path to the dataset. This should be a directory containing
                the following files:
                - `images/`: A directory containing the images. This directory
                    should be further split into a `train/` and `test/` directory.
                - `{split}.json`: A JSON file for each split containing the
                    annotations for the samples in the split.
                - `categories.json`: A JSON file containing the names of the entity
                    classes.
                - `verbs.json`: A JSON file containing the names of the interaction
                    classes.
            split: The split of the dataset to load. At the moment, only the `train` and
                `test` splits are supported.
        """
        self._path = Path(path)
        self._split = split

        self._samples = self._get_samples()
        self._entity_class_to_id = self._get_entity_classes()
        self._interaction_class_to_id = self._get_interaction_classes()

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def num_interaction_classes(self) -> int:
        return len(self._interaction_class_to_id)

    # ---------------------------------------------------------------------- #
    # Public Methods
    # ---------------------------------------------------------------------- #

    def get_interactions_per_entity(self) -> List[List[int]]:
        with open(self._path / "interaction_tuples.json") as f:
            tuples = json.load(f)

        object_valid_interactions = [[] for _ in range(len(self._entity_class_to_id))]
        for tuple_ in tuples:
            object_id = self._entity_class_to_id[tuple_["object"]]
            interaction_id = self._interaction_class_to_id[tuple_["interaction"]]

            object_valid_interactions[object_id].append(interaction_id)

        return object_valid_interactions

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self._samples[idx]

        image_path = self._path / "images" / self._split / f"{sample['id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        entity_coordinates = []
        entity_labels = []

        for entity in sample["entities"]:
            entity_coordinates.append(entity["bbox"])
            entity_labels.append(self._entity_class_to_id[entity["class"]])

        entity_coordinates = torch.as_tensor(entity_coordinates, dtype=torch.float32)
        entity_labels = torch.as_tensor(entity_labels, dtype=torch.int64)

        # coordinates are normalized to [0, 1], but we need them in [0, W] and [0, H]
        entity_coordinates[:, [0, 2]] *= image.width
        entity_coordinates[:, [1, 3]] *= image.height

        inter_indices = []
        inter_classes = []

        h2o_inter_indices = torch.empty(
            (len(sample["interactions"]), 2), dtype=torch.long
        )
        h2o_inter_labels = torch.zeros(
            (len(sample["interactions"]), len(self._interaction_class_to_id)),
            dtype=torch.float,
        )

        for idx, interaction in enumerate(sample["interactions"]):
            subject_idx = interaction["subject_idx"]
            object_idx = interaction["object_idx"]
            h2o_inter_indices[idx, 0] = subject_idx
            h2o_inter_indices[idx, 1] = object_idx

            for inter_class in interaction["classes"]:
                inter_indices.append([subject_idx, object_idx])
                inter_classes.append(self._interaction_class_to_id[inter_class])

            for class_ in interaction["classes"]:
                h2o_inter_labels[idx, self._interaction_class_to_id[class_]] = 1.0

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

        target = {
            "boxes_h": boxes_h,
            "boxes_o": boxes_o,
            "labels": inter_classes,
            "object": objects,
            "h2o_entity_boxes": entity_coordinates,
            "h2o_entity_labels": entity_labels,
            "h2o_interaction_indices": h2o_inter_indices,
            "h2o_interaction_labels": h2o_inter_labels,
        }

        return image, target

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    def _get_samples(self) -> List[FileSample]:
        with open(self._path / f"{self._split}.json") as file:
            data = json.load(file)

        # In H2O all samples have at least one entity
        return data

    def _get_entity_classes(self) -> Dict[str, int]:
        with open(self._path / "entity_classes.json") as file:
            classes: list[str] = json.load(file)

        return {name: i for i, name in enumerate(classes)}

    def _get_interaction_classes(self) -> Dict[str, int]:
        with open(self._path / "interaction_classes.json") as file:
            classes: list[str] = json.load(file)

        return {name: i for i, name in enumerate(classes)}

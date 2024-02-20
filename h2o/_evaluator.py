##
##
##

from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

import torch
from torchmetrics.classification import MultilabelAveragePrecision
from torchvision import ops


class H2OEvaluator:
    """Evaluator for the Human-Object Interaction (HOI) task.

    The evaluator computes the accuracy and mean average precision (mAP) for the
    predictions.
    """

    def __init__(
        self,
        num_interaction_classes: int,
        iou_threshold: float = 0.5,
        map_thresholds: Union[int, List[float], None] = None,
    ) -> None:
        """Initializes a new evaluator.

        Args:
            num_interaction_classes: The number of interaction classes.
            iou_threshold: Only the interaction pairs with an IoU greater than this
                threshold are considered as potential matches.
            average: The averaging strategy to use for the metrics.
            map_thresholds: The thresholds to use for the computation of the mean
                average precision.
        """
        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError(
                f"The IoU threshold must be in the range [0, 1], got {iou_threshold}."
            )

        self.iou_threshold = iou_threshold

        self._map = MultilabelAveragePrecision(
            num_classes=num_interaction_classes,
            threshold=map_thresholds,
            average="none",
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def update(
        self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> None:
        pred, tgt = _match_prediction_gold(output, target, self.iou_threshold)
        self._map.update(pred, tgt)

    def compute(self) -> torch.Tensor:
        return self._map.compute()

    def reset(self) -> None:
        self._map.reset()

    def to(self, device: Union[torch.device, str]) -> "H2OEvaluator":
        self._map.to(device)
        return self


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MatchIndices:
    matched_predictions: torch.Tensor
    matched_targets: torch.Tensor
    not_matched_predictions: torch.Tensor
    not_matched_targets: torch.Tensor


def _match_prediction_gold(
    pred: Dict[str, torch.Tensor],
    gold: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = _get_match_indices(pred, gold, iou_threshold)
    pred_labels = pred["interaction_labels"]
    gold_labels = gold["interaction_labels"]

    total = (
        len(indices.matched_predictions)
        + len(indices.not_matched_predictions)
        + len(indices.not_matched_targets)
    )
    num_classes = pred_labels.shape[1]

    predictions = pred_labels.new_zeros((total, num_classes))
    targets = pred_labels.new_zeros((total, num_classes))

    # MATCHED PRED - MATCHED TARGET
    # NOT MATCHED PRED - fake target
    # fake pred - NOT MATCHED TARGET

    predictions[: len(indices.matched_predictions)] = pred_labels[
        indices.matched_predictions
    ]
    targets[: len(indices.matched_targets)] = gold_labels[indices.matched_targets]

    predictions[
        len(indices.matched_predictions) : len(indices.matched_predictions)
        + len(indices.not_matched_predictions)
    ] = pred_labels[indices.not_matched_predictions]

    targets[
        len(indices.matched_targets) + len(indices.not_matched_predictions) :
    ] = gold_labels[indices.not_matched_targets]

    return predictions, targets


def _get_match_indices(
    pred: Dict[str, torch.Tensor],
    gold: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> MatchIndices:
    """Returns the indices of the matched interaction pairs."""
    device = pred["entity_boxes"].device
    iou_matrix = _compute_iou_matrix(pred, gold, iou_threshold)

    matched_pred = []
    matched_gold = []
    not_matched_pred = set(range(len(pred["pairing"])))
    not_matched_gold = set(range(len(gold["pairing"])))

    max_num_matches = min(*iou_matrix.shape)
    for _ in range(max_num_matches):
        if iou_matrix.count_nonzero() == 0:
            break

        max_iou, max_iou_indices = iou_matrix.max(dim=1)  # (G,)
        _, sorted_max_iou_indices = max_iou.sort(descending=True)

        gold_matched = int(sorted_max_iou_indices[0].item())
        pred_matched = int(max_iou_indices[gold_matched].item())

        matched_pred.append(pred_matched)
        matched_gold.append(gold_matched)
        not_matched_pred.remove(pred_matched)
        not_matched_gold.remove(gold_matched)

        iou_matrix[:, pred_matched] = 0
        iou_matrix[gold_matched, :] = 0

    return MatchIndices(
        matched_predictions=torch.as_tensor(
            matched_pred, device=device, dtype=torch.long
        ),
        matched_targets=torch.as_tensor(matched_gold, device=device, dtype=torch.long),
        not_matched_predictions=torch.as_tensor(
            list(not_matched_pred), device=device, dtype=torch.long
        ),
        not_matched_targets=torch.as_tensor(
            list(not_matched_gold), device=device, dtype=torch.long
        ),
    )


def _compute_iou_matrix(
    pred: Dict[str, torch.Tensor],
    gold: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> torch.Tensor:
    pred_boxes = pred["entity_boxes"]
    gold_boxes = gold["entity_boxes"]
    pred_indices = pred["interaction_indices"]
    gold_indices = gold["interaction_indices"]

    sub_pred_boxes = pred_boxes[pred_indices[:, 0]]
    obj_pred_boxes = pred_boxes[pred_indices[:, 1]]

    sub_gold_boxes = gold_boxes[gold_indices[:, 0]]
    obj_gold_boxes = gold_boxes[gold_indices[:, 1]]

    sub_iou_matrix = ops.box_iou(sub_gold_boxes, sub_pred_boxes)
    obj_iou_matrix = ops.box_iou(obj_gold_boxes, obj_pred_boxes)
    iou_matrix = torch.min(sub_iou_matrix, obj_iou_matrix)
    iou_matrix[iou_matrix < iou_threshold] = 0

    # Set the IoU of pairs where the predicted labels are different from the
    # ground truth labels (of the corresponding entities in the pair) to 0.
    sub_pred_labels = pred["entity_labels"][pred_indices[:, 0]]
    obj_pred_labels = pred["entity_labels"][pred_indices[:, 1]]

    sub_gold_labels = gold["entity_labels"][gold_indices[:, 0]]
    obj_gold_labels = gold["entity_labels"][gold_indices[:, 1]]

    sub_label_matrix = sub_gold_labels.unsqueeze(1) != sub_pred_labels  # (G, P)
    obj_label_matrix = obj_gold_labels.unsqueeze(1) != obj_pred_labels  # (G, P)
    label_matrix = sub_label_matrix.logical_or_(obj_label_matrix)  # (G, P)
    iou_matrix[label_matrix] = 0

    return iou_matrix

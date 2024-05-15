##
##
##

from typing import TypedDict

from PIL import Image
from torch import Tensor
from typing_extensions import TypeAlias

Sample: TypeAlias = Image.Image


class Annotation(TypedDict):
    boxes_h: Tensor
    boxes_o: Tensor
    object: Tensor
    labels: Tensor
    custom_entity_boxes: Tensor
    custom_entity_labels: Tensor
    custom_interaction_indices: Tensor
    custom_interaction_labels: Tensor

##
##
##

from typing import List, Optional, Tuple

from typing_extensions import Protocol

from ._types import Annotation, Sample


class Dataset(Protocol):
    """Interface for Human-Object Interaction Detection datasets."""

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def human_class_index(self) -> int:
        """Returns the index of the human class."""
        ...

    def entity_classes(self) -> Tuple[str, ...]:
        """Returns the list of entity classes.

        The classes are ordered by their index in the dataset.
        """
        ...

    def interaction_classes(self) -> Tuple[str, ...]:
        """Returns the list of interaction classes.

        The classes are ordered by their index in the dataset.
        """
        ...

    def valid_interaction_tuples(self) -> Optional[List[List[int]]]:
        """Returns for each entity the list of interactions in which it can be involved.

        Returns:
            A list of lists, where the i-th list contains the indices of the
            interactions in which the i-th entity can be involved. If such information
            is not available, `None` is returned.
        """
        ...

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Tuple[Sample, Annotation]: ...

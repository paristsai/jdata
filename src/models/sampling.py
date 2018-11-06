from abc import ABC, abstractmethod
from imblearn.under_sampling import (
    AllKNN,
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)


class Sampler(ABC):
    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def fit_sample(self, X, y, *args):
        raise NotImplementedError


# TODO: rewrite
def sampler(name, ratio, random_state=0, return_indices=True, **kwargs):
    if name == "rus":
        sampler = RandomUnderSampler(
            ratio=ratio,
            return_indices=return_indices,
            random_state=random_state,
            **kwargs,
        )
    elif name == "nm":
        sampler = NearMiss(
            ratio=ratio,
            return_indices=return_indices,
            random_state=random_state,
            **kwargs,
        )
    elif name == "enn":
        sampler = EditedNearestNeighbours(
            return_indices=return_indices, random_state=random_state, **kwargs
        )
    elif name == "renn":
        sampler = RepeatedEditedNearestNeighbours(
            return_indices=return_indices, random_state=random_state, **kwargs
        )
    elif name == "allknn":
        sampler = AllKNN(
            return_indices=return_indices, random_state=random_state, **kwargs
        )
    elif name == "tl":
        sampler = TomekLinks(
            return_indices=return_indices, random_state=random_state, **kwargs
        )
    else:
        raise ValueError
    return sampler

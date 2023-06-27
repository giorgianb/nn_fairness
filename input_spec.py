from collections import namedtuple
from dataclasses import dataclass
import dataclasses
from collections.abc import Sequence
import numpy as np
from itertools import product
import warnings

@dataclass(frozen=True)
class Continuous:
    min: float
    max: float
    label: str = None

@dataclass(frozen=True)
class Discrete:
    values: Sequence
    label: str = None

@dataclass(frozen=True)
class OneHot:
    label: str 

@dataclass(frozen=True)
class InputClass:
    boxes: Sequence[np.ndarray]

    def __and__(self, other):
        boxes = []
        for box1, box2 in product(self.boxes, other.boxes):
            intersection = np.concatenate((np.maximum(box1[:, 0], box2[:, 0])[:, None], np.minimum(box1[:, 1], box2[:, 1])[:, None]), axis=1)
            if np.all(intersection[:, 0] <= intersection[:, 1]):
                boxes.append(intersection)

        return InputClass(boxes)

    def __or__(self, other):
        # TODO: Implement mutually exclusive regions. This will involve splitting the boxes
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(other.boxes)):
                intersection = np.concatenate((np.maximum(self.boxes[i][:, 0], other.boxes[j][:, 0])[:, None], np.minimum(self.boxes[i][:, 1], other.boxes[j][:, 1])[:, None]), axis=1)
                if np.all(intersection[:, 0] <= intersection[:, 1]):
                    warnings.warn("regions are not mutually exclusive")

        return InputClass(tuple(self.boxes) + tuple(other.boxes))

    def __not__(self):
        raise NotImplementedError("Need to know the domain to implement this")

    def __iter__(self):
        return iter(self.boxes)

class InputSpecification:
    def __init__(self, *variables):
        self.variables = {}
        self.one_hot_variables = {}
        self.discrete_indices = {}
        self.continuous_indices = {}
        self.n_variables = len(variables)

        for i, v in enumerate(variables):
            if v.label is None:
                v = dataclasses.replace(v, label=f'x_{i}')

            if isinstance(v, Continuous):
                self.variables[v.label] = v
                self.continuous_indices[v.label] = i
            elif isinstance(v, Discrete):
                self.variables[v.label] = v
                self.discrete_indices[v.label] = i
            else:
                if v.label not in self.one_hot_variables:
                    self.one_hot_variables[v.label] = []

                self.one_hot_variables[v.label].append(i)

    def get(self, **kwargs):
        box = np.zeros((self.n_variables, 2), dtype=np.float32)
        for k, v in self.variables.items():
            if isinstance(v, Continuous):
                box[self.continuous_indices[k]] = [v.min, v.max]
            elif isinstance(v, Discrete):
                box[self.discrete_indices[k]] = [min(v.values), max(v.values)]

        one_hot_specification = dict(self.one_hot_variables.items())
        for k, v in kwargs.items():
            # TODO: add checking for valid values
            if k in self.continuous_indices:
                box[self.continuous_indices[k]] = v
            elif k in self.discrete_indices:
                box[self.discrete_indices[k]] = v
            elif k in one_hot_specification:
                one_hot_specification[k] = (self.one_hot_variables[k][v],)
            else:
                raise ValueError(f'Invalid variable name {k}')

        boxes = []
        for hot_values in product(*one_hot_specification.values()):
            box_copy = box.copy()
            if len(hot_values) > 0:
                box_copy[hot_values] = 1
            boxes.append(box_copy)

        return InputClass(tuple(boxes))

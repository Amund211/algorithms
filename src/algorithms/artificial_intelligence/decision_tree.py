"""
function D ECISION -T REE -L EARNING (examples, attributes, parent examples) returns
a tree
if examples is empty then return P LURALITY-VALUE (parent examples)
else if all examples have the same classification then return the classification
else if attributes is empty then return P LURALITY-VALUE (examples)
else
A ← argmaxa ∈ attributes I MPORTANCE(a, examples)
tree ← a new decision tree with root test A
for each value vk of A do
exs ← {e : e ∈ examples and e.A = vk}
subtree ← D ECISION -T REE -L EARNING (exs, attributes − A, examples)
add a branch to tree with label (A = vk) and subtree subtree
return tree
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable, Collection, Generic, Iterable, Mapping, Set, TypeVar

AttributeName = TypeVar("AttributeName")
AttributeValue = TypeVar("AttributeValue")
Classification = TypeVar("Classification")

RateImportance = Callable[
    [
        AttributeName,
        Collection[tuple[Mapping[AttributeName, AttributeValue], Classification]],
    ],
    float,
]


@dataclass
class DTInternalNode(Generic[AttributeName, AttributeValue, Classification]):
    attribute: AttributeName
    children: Mapping[
        AttributeValue, DTNode[AttributeName, AttributeValue, Classification]
    ]


@dataclass
class DTLeafNode(Generic[Classification]):
    classification: Classification


DTNode = (
    DTInternalNode[AttributeName, AttributeValue, Classification]
    | DTLeafNode[Classification]
)


def get_classifications(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], Classification]]
) -> Iterable[Classification]:
    """Extract the classifications from the data"""
    return (classification for example, classification in data)


def plurality_value(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], Classification]]
) -> Classification:
    """Return the most common classification in the dataset"""
    element, frequency = collections.Counter(get_classifications(data)).most_common(1)[
        0
    ]
    return element


def learn_decision_tree(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], Classification]],
    importance: RateImportance[AttributeName, AttributeValue, Classification],
    values: Collection[AttributeValue],
) -> DTNode[AttributeName, AttributeValue, Classification]:
    pass


def _learn_decision_tree(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], Classification]],
    parent_data: Collection[
        tuple[Mapping[AttributeName, AttributeValue], Classification]
    ],
    attributes: Set[AttributeName],
    importance: RateImportance[AttributeName, AttributeValue, Classification],
    values: Collection[AttributeValue],
) -> DTNode[AttributeName, AttributeValue, Classification]:
    if not data:
        return DTLeafNode(plurality_value(parent_data))
    elif len(set(get_classifications(data))) == 1:
        return DTLeafNode(next(iter(get_classifications(data))))
    elif not attributes:
        return DTLeafNode(plurality_value(data))

    chosen_attribute = max(
        attributes, key=lambda attribute: importance(attribute, data)
    )

    return DTInternalNode(
        attribute=chosen_attribute,
        children={
            value: _learn_decision_tree(
                data=tuple(
                    filter(lambda element: element[0][chosen_attribute] == value, data)
                ),
                parent_data=data,
                attributes=attributes - {chosen_attribute},
                importance=importance,
                values=values,
            )
            for value in values
        },
    )

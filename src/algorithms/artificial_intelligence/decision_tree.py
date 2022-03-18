from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Set,
    TypeVar,
)

if TYPE_CHECKING:
    import graphviz  # type: ignore

AttributeName = TypeVar("AttributeName")
AttributeValue = TypeVar("AttributeValue")

RateImportance = Callable[
    [
        AttributeName,
        Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
        Collection[AttributeValue],
    ],
    float,
]


@dataclass
class DTInternalNode(Generic[AttributeName, AttributeValue]):
    attribute: AttributeName
    children: Mapping[AttributeValue, DTNode[AttributeName, AttributeValue]]


@dataclass
class DTLeafNode:
    classification: bool


DTNode = DTInternalNode[AttributeName, AttributeValue] | DTLeafNode


def get_classifications(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]]
) -> Iterable[bool]:
    """Extract the classifications from the data"""
    return (classification for example, classification in data)


def get_distribution(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]]
) -> tuple[float, float]:
    """Return the amount of negative and positive examples in the set"""
    amt_examples = len(data)
    amt_positive = sum(get_classifications(data))
    amt_negative = amt_examples - amt_positive
    return amt_negative, amt_positive


def random_importance(
    attribute: AttributeName,
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
    values: Collection[AttributeValue],
) -> float:
    """Randomly select an attribute to split over"""
    return random.random()


def bernoulli_entropy(p: float) -> float:
    """Compute the entropy of a Bernoulli distribution with probability p"""
    if p == 0 or p == 1:
        return 0
    assert 0 < p < 1
    assert 0 < 1 - p < 1
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def information_gain_importance(
    attribute: AttributeName,
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
    values: Collection[AttributeValue],
) -> float:
    """Split on the attribute with the largest information gain"""
    n, p = get_distribution(data)

    def branch_remainder(
        data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
        value: AttributeValue,
    ) -> float:
        """
        Compute the contribution of one branch to the remaining entropy calculation
        """
        remaining_data = tuple(
            filter(lambda example: example[0][attribute] == value, data)
        )
        if not remaining_data:
            return 0
        n_k, p_k = get_distribution(remaining_data)
        return (p_k + n_k) / (p + n) * bernoulli_entropy(p_k / (p_k + n_k))

    return bernoulli_entropy(p / (p + n)) - sum(
        branch_remainder(data, value) for value in values
    )


def plurality_value(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]]
) -> bool:
    """Return the most common classification in the dataset"""
    element, frequency = collections.Counter(get_classifications(data)).most_common(1)[
        0
    ]
    return element


def learn_decision_tree(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
    importance: RateImportance[AttributeName, AttributeValue],
    values: Collection[AttributeValue],
) -> DTNode[AttributeName, AttributeValue]:
    """Learn a decision tree for boolean classification"""
    if not data:
        raise ValueError("No data to train on")

    attributes = set(next(iter(data))[0])
    for example, classification in data:
        assert attributes == set(example)

    return _learn_decision_tree(
        data=data,
        parent_data=data,
        attributes=attributes,
        importance=importance,
        values=values,
    )


def _learn_decision_tree(
    data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
    parent_data: Collection[tuple[Mapping[AttributeName, AttributeValue], bool]],
    attributes: Set[AttributeName],
    importance: RateImportance[AttributeName, AttributeValue],
    values: Collection[AttributeValue],
) -> DTNode[AttributeName, AttributeValue]:
    if not data:
        return DTLeafNode(plurality_value(parent_data))
    elif len(set(get_classifications(data))) == 1:
        return DTLeafNode(next(iter(get_classifications(data))))
    elif not attributes:
        return DTLeafNode(plurality_value(data))

    chosen_attribute = max(
        attributes, key=lambda attribute: importance(attribute, data, values)
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


def classify(
    root: DTNode[AttributeName, AttributeValue],
    example: Mapping[AttributeName, AttributeValue],
) -> bool:
    """Classify an example using the given decision tree"""
    if isinstance(root, DTLeafNode):
        return root.classification

    return classify(root.children[example[root.attribute]], example)


def visualize_decision_tree(
    root: DTNode[AttributeName, AttributeValue], outfile: Path
) -> None:
    """
    Use graphviz to create a pdf of the decision tree

    Requires installing with the [visualize] extra (graphviz)
    """
    import graphviz

    visualization_tree = graphviz.Digraph(name="Decision Tree")

    _add_node_to_visualization(visualization_tree, root, parent_name="", value=None)

    visualization_tree.render(outfile=outfile)


def _add_node_to_visualization(
    visualization_tree: graphviz.Digraph,
    node: DTNode[AttributeName, AttributeValue],
    parent_name: str,
    value: AttributeValue | None,
) -> None:
    """Recursively add this node and its children to the graph"""

    if isinstance(node, DTInternalNode):
        own_name = f"{node.attribute}="
    else:
        own_name = "leaf"

    if value is None:
        # Root node
        name = f"root-{own_name}"
    else:
        name = f"{parent_name}{value}-{own_name}"

    visualization_tree.node(
        name=name,
        label=f"Attr. {own_name[:-1]}"
        if isinstance(node, DTInternalNode)
        else str(node.classification),
    )

    if value is not None:
        # The root node has no parent
        visualization_tree.edge(tail_name=parent_name, head_name=name, label=str(value))

    if isinstance(node, DTInternalNode):
        for value, child in node.children.items():
            _add_node_to_visualization(
                visualization_tree, child, parent_name=name, value=value
            )


if __name__ == "__main__":
    import csv

    BinaryValue = Literal[1, 2]
    values: tuple[BinaryValue, BinaryValue] = (1, 2)
    ExampleData = Collection[tuple[Mapping[BinaryValue, BinaryValue], bool]]

    def read_data(path: Path) -> ExampleData:
        """Read the binary attribute data from the path"""
        with path.open(newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=",")

            raw_data = tuple(reader)

        unstructured_data = map(lambda raw_row: tuple(map(int, raw_row)), raw_data)

        # mypy struggles with narrowing to literal types, so we type ignore this later
        assert all(all(value in values for value in row) for row in unstructured_data)

        return tuple(
            (
                {i: row[i] for i in range(len(row) - 1)},  # type: ignore
                row[-1] == 2,  # 1 -> False, 2 -> True
            )
            for row in map(lambda raw_row: tuple(map(int, raw_row)), raw_data)
        )

    train_data = read_data(Path("train.csv"))
    test_data = read_data(Path("test.csv"))

    random_tree = learn_decision_tree(
        data=train_data, importance=random_importance, values=values
    )

    information_gain_tree = learn_decision_tree(
        data=train_data, importance=information_gain_importance, values=values
    )

    visualize_decision_tree(random_tree, outfile=Path("random_tree.png"))
    visualize_decision_tree(
        information_gain_tree, outfile=Path("information_gain_tree.png")
    )

    total_examples = len(test_data)
    random_correct_examples = 0
    information_gain_correct_examples = 0
    for example, classification in test_data:
        random_correct_examples += classify(random_tree, example) == classification
        information_gain_correct_examples += (
            classify(information_gain_tree, example) == classification
        )

    print(
        "Accuracy using random attribute selection: "
        f"{random_correct_examples/total_examples*100:.2f}%"
    )
    print(
        "Accuracy using information gain attribute selection: "
        f"{information_gain_correct_examples/total_examples*100:.2f}%"
    )

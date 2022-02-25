from enum import Enum, unique
from functools import partial
from typing import Mapping, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

State = TypeVar("State", bound=int)
Evidence = TypeVar("Evidence", bound=int)


def forward(
    prior: Mapping[State, float],
    evidence: Evidence,
    dynamic_model: Mapping[State, Mapping[State, float]],
    observation_model: Mapping[State, Mapping[Evidence, float]],
) -> Mapping[State, float]:
    """
    Calculate the new state distribution given the new evidence

    prior: mapping state->probability
    evidence: the new evidence
    dynamic_model: mapping current state -> (mapping new state -> probability)
    observation_model: mapping state -> (mapping evidence -> probability)
    """

    assert len(prior) == len(dynamic_model) == len(observation_model) > 0

    # P(e_{t+1} | X_{t+1})
    P_etp1_given_Xtp1: npt.NDArray[np.float64] = np.array(
        [
            probability_of_evidence[evidence]
            for state, probability_of_evidence in observation_model.items()
        ],
        dtype=np.float64,
    )

    # ∑_{x_t} P(X_{t+1} | x_t) P(x_t | e_{1:t})
    # Type ignored because default for sum is 0 (int)
    S: npt.NDArray[np.float64] = sum(
        np.array(
            [
                prior[old_state] * probability
                for new_state, probability in probability_of_new_state.items()
            ],
            dtype=np.float64,
        )
        for old_state, probability_of_new_state in dynamic_model.items()
    )  # type: ignore

    # P(X_{t+1} | e_{1:t+1}) = α P(e_{t+1} | X_{t+1}) ∑_{x_t} P(X_{t+1} | x_t) P(x_t | e_{1:t})  # noqa: E501
    non_normalized = P_etp1_given_Xtp1 * S

    normalized = non_normalized / non_normalized.sum()

    # Map the probabilities out to our format
    return {state: probability for state, probability in zip(dynamic_model, normalized)}


def backward(
    p: Mapping[State, float],
    evidence: Evidence,
    dynamic_model: Mapping[State, Mapping[State, float]],
    observation_model: Mapping[State, Mapping[Evidence, float]],
) -> Mapping[State, float]:
    """
    Calculate the new state distribution given the new evidence

    p: mapping state->"probability"
    evidence: the old evidence
    dynamic_model: mapping current state -> (mapping new state -> probability)
    observation_model: mapping state -> (mapping evidence -> probability)
    """

    assert len(p) == len(dynamic_model) == len(observation_model) > 0

    # P(e_{k+1:t} | X_k) = ∑_{x_{k+1}} P(e_{k+1} | x_{k+1}) P(e_{k+2:t} | x_{k+1}) P(x_{k+1} | X_k)  # noqa: E501

    S: npt.NDArray[np.float64] = sum(
        # P(e_{k+1} | x_{k+1})
        observation_model[old_state][evidence]  # type: ignore
        # P(e_{k+2:t} | x_{k+1})
        * p[old_state]
        # P(x_{k+1} | X_k)
        * np.array(
            [probability for probability in probability_of_new_state.values()],
            dtype=np.float64,
        )
        for old_state, probability_of_new_state in dynamic_model.items()
    )

    # Map the probabilities out to our format
    return {state: probability for state, probability in zip(dynamic_model, S)}


def filtering(
    prior: Mapping[State, float],
    evidences: Sequence[Evidence],
    dynamic_model: Mapping[State, Mapping[State, float]],
    observation_model: Mapping[State, Mapping[Evidence, float]],
) -> list[Mapping[State, float]]:
    """Calculate all the forward messages sequentially"""
    return [
        (
            prior := forward(
                prior=prior,
                evidence=evidence,
                dynamic_model=dynamic_model,
                observation_model=observation_model,
            )
        )
        for evidence in evidences
    ]


def make_backward_messages(
    p: Mapping[State, float],
    evidences: Sequence[Evidence],
    dynamic_model: Mapping[State, Mapping[State, float]],
    observation_model: Mapping[State, Mapping[Evidence, float]],
) -> list[Mapping[State, float]]:
    """Calculate all the backward messages sequentially"""
    return [
        (
            p := backward(
                p=p,
                evidence=evidence,
                dynamic_model=dynamic_model,
                observation_model=observation_model,
            )
        )
        for evidence in reversed(evidences)
    ]


def smoothing(
    prior: Mapping[State, float],
    evidences: Sequence[Evidence],
    dynamic_model: Mapping[State, Mapping[State, float]],
    observation_model: Mapping[State, Mapping[Evidence, float]],
) -> tuple[list[Mapping[State, float]], list[Mapping[State, float]]]:
    forward_messages = filtering(
        prior=prior,
        evidences=evidences,
        dynamic_model=dynamic_model,
        observation_model=observation_model,
    )

    backward_messages = make_backward_messages(
        p={state: 1 for state in dynamic_model},
        evidences=evidences,
        dynamic_model=dynamic_model,
        observation_model=observation_model,
    )

    return forward_messages, backward_messages


@unique
class UmbrellaState(int, Enum):
    """The state space in the umbrella world"""

    CLEAR = 0
    RAIN = 1


@unique
class UmbrellaEvidence(int, Enum):
    """The evidence space in the umbrella world"""

    NO_UMBRELLA = 0
    UMBRELLA = 1


INITIAL_UMBRELLA_STATE = {
    UmbrellaState.CLEAR: 0.5,
    UmbrellaState.RAIN: 0.5,
}  # Common prior state

# Apply the probability distributions for the dynamic- and observation-models
# for the umbrella world
filter_umbrella = partial(
    filtering,
    INITIAL_UMBRELLA_STATE,
    dynamic_model={
        UmbrellaState.CLEAR: {UmbrellaState.CLEAR: 0.7, UmbrellaState.RAIN: 0.3},
        UmbrellaState.RAIN: {UmbrellaState.CLEAR: 0.3, UmbrellaState.RAIN: 0.7},
    },
    observation_model={
        UmbrellaState.CLEAR: {
            UmbrellaEvidence.NO_UMBRELLA: 0.8,
            UmbrellaEvidence.UMBRELLA: 0.2,
        },
        UmbrellaState.RAIN: {
            UmbrellaEvidence.NO_UMBRELLA: 0.1,
            UmbrellaEvidence.UMBRELLA: 0.9,
        },
    },
)
smoothing_umbrella = partial(
    smoothing,
    INITIAL_UMBRELLA_STATE,
    dynamic_model={
        UmbrellaState.CLEAR: {UmbrellaState.CLEAR: 0.7, UmbrellaState.RAIN: 0.3},
        UmbrellaState.RAIN: {UmbrellaState.CLEAR: 0.3, UmbrellaState.RAIN: 0.7},
    },
    observation_model={
        UmbrellaState.CLEAR: {
            UmbrellaEvidence.NO_UMBRELLA: 0.8,
            UmbrellaEvidence.UMBRELLA: 0.2,
        },
        UmbrellaState.RAIN: {
            UmbrellaEvidence.NO_UMBRELLA: 0.1,
            UmbrellaEvidence.UMBRELLA: 0.9,
        },
    },
)


def task_2_2() -> None:
    messages_list = filter_umbrella(
        [UmbrellaEvidence.UMBRELLA, UmbrellaEvidence.UMBRELLA]
    )
    # mypy is dumb and doesn't understand the index type
    print(
        "Probability of rain after observed (umbrella, umbrella): "
        f"{messages_list[-1][UmbrellaState.RAIN]}"  # type: ignore
    )

    evidences = [
        UmbrellaEvidence.UMBRELLA,
        UmbrellaEvidence.UMBRELLA,
        UmbrellaEvidence.NO_UMBRELLA,
        UmbrellaEvidence.UMBRELLA,
        UmbrellaEvidence.UMBRELLA,
    ]
    messages_list = filter_umbrella(evidences)

    # mypy is dumb and doesn't understand the index type
    print(
        "Probability of rain after observed "
        "(umbrella, umbrella, no umbrella, umbrella, umbrella): "
        f"{messages_list[-1][UmbrellaState.RAIN]}"  # type: ignore
    )
    print("Normalized forward messages:")
    print(f"\t{'(initial)'.ljust(12)} -> {INITIAL_UMBRELLA_STATE}")
    for evidence, messages in zip(evidences, messages_list, strict=True):
        print(f"\t{evidence.name.ljust(12)} -> {messages}")


def task_2_3() -> None:
    forward_messages, backward_messages = smoothing_umbrella(
        [UmbrellaEvidence.UMBRELLA, UmbrellaEvidence.UMBRELLA]
    )
    print(
        np.array(list(forward_messages[0].values()))
        * np.array(list(backward_messages[0].values()))
    )

    # Ran out of time for this task

    return


def main() -> None:
    task_2_2()
    # task_2_3()


if __name__ == "__main__":
    main()

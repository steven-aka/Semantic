from src.data.build_v7_expanded_train import select_all_attainable
from src.data.schemas import QAExample


def _example(index: int) -> QAExample:
    return QAExample(
        example_id=f"example-{index}",
        dataset="synthetic",
        question="question",
        answer="answer",
        context="context",
    )


def test_selects_all_attainable_and_preserves_frozen_prefix() -> None:
    candidates = [_example(index) for index in range(6)]
    maximum = {
        "example-0": 0.95,
        "example-1": 0.50,
        "example-2": 0.90,
        "example-3": 0.89,
        "example-4": 1.00,
        "example-5": 0.91,
    }
    selected = select_all_attainable(
        candidates,
        maximum,
        ["example-0", "example-2"],
        minimum_fidelity=0.90,
    )
    assert [row.example_id for row in selected] == [
        "example-0",
        "example-2",
        "example-4",
        "example-5",
    ]


def test_rejects_parent_that_is_not_attainable_prefix() -> None:
    candidates = [_example(index) for index in range(3)]
    maximum = {row.example_id: 1.0 for row in candidates}
    try:
        select_all_attainable(
            candidates,
            maximum,
            ["example-0", "example-2"],
            minimum_fidelity=0.90,
        )
    except ValueError as error:
        assert "prefix" in str(error)
    else:
        raise AssertionError("non-prefix parent selection was accepted")

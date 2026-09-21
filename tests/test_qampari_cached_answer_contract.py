from src.evaluation.qampari_metrics import (
    parse_cached_list_prediction,
    parse_list_prediction,
    qampari_list_metrics,
)


def test_cached_proper_name_is_not_reparsed_as_boolean():
    cached = "No. 17 Squadron # Example"
    assert parse_list_prediction(cached) == ["no"]  # Raw-output parser semantics.
    answers = parse_cached_list_prediction(cached)
    assert answers == ["No. 17 Squadron", "Example"]
    atoms = [{"answer_text": "No. 17 Squadron"}, {"answer_text": "Example"}]
    assert qampari_list_metrics(answers, atoms)["f1"] == 1.0


def test_cached_and_raw_parsing_agree_on_ordinary_list():
    value = "Alpha # Beta"
    assert parse_cached_list_prediction(value) == parse_list_prediction(value)

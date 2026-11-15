import pytest
import torch

from distill import ResidualMLP, gate, load_data


def test_model_accepts_only_current_observation():
    model = ResidualMLP(torch.zeros(43), torch.ones(43))
    action = model(torch.ones((4, 43)))
    assert action.shape == (4, 3)
    assert torch.all(action.abs() <= 1)
    with pytest.raises(RuntimeError):
        model(torch.ones((4, 40)))


def test_frozen_data_splits_and_normalized_labels():
    groups = load_data()
    assert set(groups) == {"train", "validation"}
    assert sum(len(d[0]) for d in groups.values()) == 33679
    for obs, labels, phases in groups.values():
        assert obs.shape == (len(phases), 43)
        assert labels.shape == (len(obs), 3)
        assert torch.isfinite(obs).all() and torch.isfinite(labels).all()
        assert torch.all(labels.abs() <= 1)


def test_gate_distinguishes_preservation_rescue_and_contact():
    rows = []
    for index in range(42):
        rows.append(dict(profile="fixed" if index >= 40 else "35mm",
                         frozen=dict(is_success=index < 34 or index >= 40, collision=None),
                         residual=dict(is_success=index not in (0, 36, 37, 38, 39, 41),
                                       collision="table" if index == 0 else None)))
    assert gate(rows) == dict(preserved=33, nominal=1, rescued=2, new_contact=1)

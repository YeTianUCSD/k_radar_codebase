import torch

from factorized_mlp.model import FactorizedGatedMLP


def test_gated_model_has_independent_normalized_attribute_gates():
    model = FactorizedGatedMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0,
    )
    assert model.gate_logits["weather"] is not model.gate_logits["road"]
    for weights in model.gate_weights().values():
        assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_forward_shapes_and_gradient_flow():
    model = FactorizedGatedMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0,
    )
    inputs = {
        "camera": torch.randn(3, 7),
        "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9),
    }
    output = model(inputs)
    assert output["logits"]["weather"].shape == (3, 6)
    assert output["logits"]["road"].shape == (3, 6)
    assert output["logits"]["lighting"].shape == (3, 2)
    sum(value.sum() for value in output["logits"].values()).backward()
    assert model.gate_logits["weather"].grad is not None

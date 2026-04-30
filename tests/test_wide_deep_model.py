"""Unit tests for the Wide & Deep architecture."""
import torch

from src.models.wide_deep import (
    WideDeepDraftNet,
    ROLE_ORDER,
    PAD_TOKEN,
    UNK_TOKEN,
)


def test_role_order_is_canonical():
    assert ROLE_ORDER == ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def test_special_tokens():
    assert PAD_TOKEN == "__PAD__"
    assert UNK_TOKEN == "__UNK__"


def test_forward_shape_full_draft():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    blue = torch.randint(2, 170, (4, 5))
    red = torch.randint(2, 170, (4, 5))
    logits = net(blue, red)
    assert logits.shape == (4,), logits.shape


def test_forward_with_pad_does_not_crash():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    # PAD id is 0 by convention; place pads in some slots
    blue = torch.tensor([[0, 5, 0, 7, 0]])
    red = torch.tensor([[3, 0, 4, 0, 6]])
    logits = net(blue, red)
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_sigmoid_in_range():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    blue = torch.randint(0, 170, (3, 5))
    red = torch.randint(0, 170, (3, 5))
    probs = torch.sigmoid(net(blue, red))
    assert (probs >= 0).all() and (probs <= 1).all()

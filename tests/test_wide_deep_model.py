"""Unit tests for the Wide & Deep architecture."""
import torch

from src.models.wide_deep import (
    LEGACY_ARCH,
    PAD_ID,
    PAD_TOKEN,
    ROLE_ORDER,
    UNK_ID,
    UNK_TOKEN,
    V2_ARCH,
    WideDeepDraftNet,
)


def test_role_order_is_canonical():
    assert ROLE_ORDER == ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def test_special_tokens():
    assert PAD_TOKEN == "__PAD__"
    assert UNK_TOKEN == "__UNK__"


def test_special_token_ids():
    assert PAD_ID == 0
    assert UNK_ID == 1


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


# ---------- v2_pairwise architecture ----------

def test_v2_pairwise_forward_shape_full_draft():
    net = WideDeepDraftNet(
        num_champions=170, embedding_dim=8, hidden_dim=16, dropout=0.0,
        architecture=V2_ARCH,
    )
    blue = torch.randint(2, 170, (4, 5))
    red = torch.randint(2, 170, (4, 5))
    logits = net(blue, red)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_v2_pairwise_pad_does_not_produce_nan():
    net = WideDeepDraftNet(
        num_champions=170, embedding_dim=8, hidden_dim=16, dropout=0.0,
        architecture=V2_ARCH,
    )
    # Edge case: completely empty draft on both sides — every slot is PAD.
    blue = torch.zeros((1, 5), dtype=torch.long)
    red = torch.zeros((1, 5), dtype=torch.long)
    logits = net(blue, red)
    assert torch.isfinite(logits).all()
    # Mixed PAD + real ids
    blue = torch.tensor([[0, 5, 0, 7, 0]])
    red = torch.tensor([[3, 0, 4, 0, 6]])
    logits = net(blue, red)
    assert torch.isfinite(logits).all()


def test_v2_pairwise_combine_concat_shape():
    net = WideDeepDraftNet(
        num_champions=50, embedding_dim=4, hidden_dim=8, dropout=0.0,
        architecture=V2_ARCH, combine="concat",
    )
    blue = torch.randint(2, 50, (3, 5))
    red = torch.randint(2, 50, (3, 5))
    logits = net(blue, red)
    assert logits.shape == (3,)


def test_legacy_and_v2_state_dicts_differ():
    """Sanity: switching architecture changes the parameter set the adapter
    needs to handle."""
    legacy = WideDeepDraftNet(
        num_champions=20, embedding_dim=4, hidden_dims=(8, 4), dropout=0.0,
        architecture=LEGACY_ARCH,
    )
    v2 = WideDeepDraftNet(
        num_champions=20, embedding_dim=4, hidden_dim=8, dropout=0.0,
        architecture=V2_ARCH,
    )
    legacy_keys = set(legacy.state_dict().keys())
    v2_keys = set(v2.state_dict().keys())
    assert "deep.embedding.weight" in legacy_keys
    assert "deep.champion_emb.weight" in v2_keys
    assert "deep.role_emb.weight" in v2_keys
    assert legacy_keys != v2_keys

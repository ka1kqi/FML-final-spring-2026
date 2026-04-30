"""End-to-end Flask route tests using the real server module.

These tests run against the *actual* loaded artifacts. They verify the API contract
without depending on whether wide_deep.pt is present (fallback path is exercised).
"""
import json

import pytest

from app.server import app, champion_list


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_get_champions_ok(client):
    resp = client.get("/api/champions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]


def test_recommend_returns_required_fields(client):
    resp = client.post(
        "/api/recommend",
        data=json.dumps({
            "blue_picks": [None] * 5,
            "red_picks": [None] * 5,
            "blue_bans": [],
            "red_bans": [],
            "step": 0,
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    rec = data["recommendations"][0]
    # backward-compat fields
    assert "champion" in rec
    assert "score" in rec
    assert "win_prob" in rec
    # new fields
    assert "performance_score" in rec
    assert "wide_deep_side_win_prob" in rec
    assert "wide_deep_blue_win_prob" in rec
    assert "final_rank_score" in rec
    assert "prob_source" in rec
    assert rec["prob_source"] in {"wide_deep", "score_heuristic_fallback"}


def test_evaluate_returns_prob_source(client):
    # Take any 10 champions from the loaded vocab
    blue = champion_list[:5]
    red = champion_list[5:10]
    resp = client.post(
        "/api/evaluate",
        data=json.dumps({"blue_picks": blue, "red_picks": red}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    # backward-compat fields
    assert "blue_win_prob" in data
    assert "red_win_prob" in data
    assert "blue_score" in data
    assert "red_score" in data
    # new fields
    assert "prob_source" in data
    assert data["prob_source"] in {
        "wide_deep", "match_classifier", "heuristic", "score_heuristic_fallback"
    }
    assert "model_version" in data

"""HTTP-level coverage for ``src/api/routes/items.py``.

These drive the real dependency graph — route, service, repository — so they
also guard the wiring in ``src/dependencies.py``.
"""

from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient


def create_item(client: TestClient, **payload: Any) -> dict[str, Any]:
    """Create an item over HTTP and return its body."""
    response = client.post("/api/v1/items", json={"name": "widget", **payload})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_returns_201_with_the_stored_representation(client: TestClient) -> None:
    response = client.post("/api/v1/items", json={"name": "widget", "description": "a thing"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "widget"
    assert body["description"] == "a thing"
    assert body["is_active"] is True
    assert UUID(body["id"])
    assert body["created_at"] == body["updated_at"]


def test_create_defaults_the_description_to_null(client: TestClient) -> None:
    assert create_item(client)["description"] is None


def test_create_rejects_unknown_fields(client: TestClient) -> None:
    """``extra="forbid"`` on the request model — a typo must not be swallowed."""
    response = client.post("/api/v1/items", json={"name": "widget", "bogus": 1})

    assert response.status_code == 422


def test_create_rejects_an_empty_name(client: TestClient) -> None:
    assert client.post("/api/v1/items", json={"name": ""}).status_code == 422


def test_create_assigns_a_distinct_id_per_item(client: TestClient) -> None:
    assert create_item(client)["id"] != create_item(client)["id"]


def test_get_returns_the_created_item(client: TestClient) -> None:
    created = create_item(client, name="findme")

    response = client.get(f"/api/v1/items/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_returns_404_for_an_unknown_id(client: TestClient) -> None:
    response = client.get(f"/api/v1/items/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_get_rejects_a_malformed_id(client: TestClient) -> None:
    assert client.get("/api/v1/items/not-a-uuid").status_code == 422


def test_list_returns_newest_first(client: TestClient) -> None:
    names = ["first", "second", "third"]
    for name in names:
        create_item(client, name=name)

    body = client.get("/api/v1/items").json()

    assert [item["name"] for item in body["items"]] == list(reversed(names))


def test_list_paginates_with_limit_and_offset(client: TestClient) -> None:
    for name in ["a", "b", "c"]:
        create_item(client, name=name)

    body = client.get("/api/v1/items", params={"limit": 1, "offset": 1}).json()

    assert [item["name"] for item in body["items"]] == ["b"]
    assert (body["limit"], body["offset"]) == (1, 1)


def test_list_reports_the_unpaginated_total(client: TestClient) -> None:
    for name in ["a", "b", "c"]:
        create_item(client, name=name)

    body = client.get("/api/v1/items", params={"limit": 1}).json()

    assert len(body["items"]) == 1
    assert body["total"] == 3


def test_list_rejects_an_out_of_range_limit(client: TestClient) -> None:
    assert client.get("/api/v1/items", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/items", params={"limit": 101}).status_code == 422


def test_list_rejects_a_negative_offset(client: TestClient) -> None:
    assert client.get("/api/v1/items", params={"offset": -1}).status_code == 422


def test_update_applies_only_the_supplied_fields(client: TestClient) -> None:
    created = create_item(client, name="before", description="kept")

    response = client.patch(f"/api/v1/items/{created['id']}", json={"name": "after"})

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "after"
    assert body["description"] == "kept"
    assert body["id"] == created["id"]
    assert body["created_at"] == created["created_at"]


def test_update_bumps_updated_at(client: TestClient) -> None:
    created = create_item(client)

    updated = client.patch(f"/api/v1/items/{created['id']}", json={"name": "renamed"}).json()

    assert updated["updated_at"] >= created["updated_at"]


def test_update_can_clear_the_nullable_description(client: TestClient) -> None:
    """``description`` is the one field an explicit null is allowed to blank."""
    created = create_item(client, description="to be cleared")

    response = client.patch(f"/api/v1/items/{created['id']}", json={"description": None})

    assert response.status_code == 200
    assert response.json()["description"] is None


def test_update_can_deactivate_an_item(client: TestClient) -> None:
    created = create_item(client)

    response = client.patch(f"/api/v1/items/{created['id']}", json={"is_active": False})

    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_update_ignores_an_explicit_null_on_a_non_nullable_field(client: TestClient) -> None:
    """Regression: a null on ``name``/``is_active`` used to blow up as a 500."""
    created = create_item(client, name="kept")

    response = client.patch(
        f"/api/v1/items/{created['id']}",
        json={"name": None, "is_active": None, "description": "changed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "kept"
    assert body["is_active"] is True
    assert body["description"] == "changed"


def test_update_rejects_a_body_of_only_nulls(client: TestClient) -> None:
    """Nothing survives the null stripping, so there is no change to apply."""
    created = create_item(client)

    response = client.patch(f"/api/v1/items/{created['id']}", json={"name": None})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_update_rejects_an_empty_body(client: TestClient) -> None:
    created = create_item(client)

    response = client.patch(f"/api/v1/items/{created['id']}", json={})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_update_rejects_unknown_fields(client: TestClient) -> None:
    created = create_item(client)

    response = client.patch(f"/api/v1/items/{created['id']}", json={"bogus": 1})

    assert response.status_code == 422


def test_update_returns_404_for_an_unknown_id(client: TestClient) -> None:
    response = client.patch(f"/api/v1/items/{uuid4()}", json={"name": "ghost"})

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_delete_returns_204_and_removes_the_item(client: TestClient) -> None:
    created = create_item(client)

    assert client.delete(f"/api/v1/items/{created['id']}").status_code == 204
    assert client.get(f"/api/v1/items/{created['id']}").status_code == 404
    assert client.get("/api/v1/items").json()["total"] == 0


def test_delete_is_not_idempotent_and_404s_the_second_time(client: TestClient) -> None:
    created = create_item(client)
    client.delete(f"/api/v1/items/{created['id']}")

    response = client.delete(f"/api/v1/items/{created['id']}")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_delete_returns_404_for_an_unknown_id(client: TestClient) -> None:
    assert client.delete(f"/api/v1/items/{uuid4()}").status_code == 404

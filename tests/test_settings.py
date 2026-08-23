"""Tests for Settings, Subscription Tiers, Custom Persona Instructions & Data Controls."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.settings import settings_manager, UserProfileUpdateRequest

client = TestClient(app)


def test_get_default_settings():
    res = client.get("/settings/profile?session_id=test_session_99")
    assert res.status_code == 200
    data = res.json()
    assert data["subscription_tier"] in ["free", "pro", "enterprise"]
    assert "custom_instructions_user" in data
    assert "theme_accent" in data


def test_update_profile_and_custom_instructions():
    update_payload = {
        "user_name": "Rishi Lead AI",
        "custom_instructions_user": "I am a Senior Staff AI Engineer specializing in distributed RAG.",
        "custom_instructions_style": "Format answers in concise mathematical LaTeX and Python.",
        "reasoning_effort": "high",
        "theme_accent": "violet",
        "auto_refine_enabled": True,
    }
    res = client.post("/settings/profile?session_id=test_session_99", json=update_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["user_name"] == "Rishi Lead AI"
    assert "Senior Staff" in data["custom_instructions_user"]
    assert data["theme_accent"] == "violet"


def test_upgrade_subscription_tier():
    # Upgrade to Pro
    res_pro = client.post("/settings/tier/upgrade?session_id=test_session_99", json={"target_tier": "pro"})
    assert res_pro.status_code == 200
    data_pro = res_pro.json()
    assert data_pro["current_tier"] == "pro"

    # Verify profile reflects Pro
    profile_res = client.get("/settings/profile?session_id=test_session_99")
    assert profile_res.json()["subscription_tier"] == "pro"

    # Downgrade to Free
    res_free = client.post("/settings/tier/upgrade?session_id=test_session_99", json={"target_tier": "free"})
    assert res_free.status_code == 200
    assert res_free.json()["current_tier"] == "free"


def test_cache_and_memory_clear_endpoints():
    # Test Cache Clear
    cache_res = client.post("/settings/cache/clear")
    assert cache_res.status_code == 200
    assert cache_res.json()["status"] == "cleared"

    # Test Memory Clear
    mem_res = client.post("/settings/memory/clear?session_id=test_session_99")
    assert mem_res.status_code == 200
    assert mem_res.json()["status"] == "cleared"


def test_export_data_archive():
    res = client.get("/settings/export?session_id=test_session_99")
    assert res.status_code == 200
    data = res.json()
    assert "export_timestamp" in data
    assert "user_settings" in data
    assert "telemetry_summary" in data

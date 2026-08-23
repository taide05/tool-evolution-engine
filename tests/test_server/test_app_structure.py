def test_app_has_no_global_conn():
    import tool_evolution.server.app as server_app
    assert not hasattr(server_app, "_conn")
    assert not hasattr(server_app, "get_db")


def test_app_routes_registered():
    from tool_evolution.server.app import app
    paths = {getattr(r, "path", "") for r in app.routes}
    for expected in ("/health", "/api/traces/report", "/api/skills/discoveries",
                     "/api/rules", "/api/analytics/summary", "/api/canary/{skill_id}/promote",
                     "/api/memory/search"):
        assert any(p == expected for p in paths)

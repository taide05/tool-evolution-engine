def test_app_has_no_global_conn():
    import tool_evolution.server.app as server_app
    assert not hasattr(server_app, "_conn")
    # get_db 允许以 import 形式存在（health 探针依赖），但不得在本模块定义
    if hasattr(server_app, "get_db"):
        assert server_app.get_db.__module__ != "tool_evolution.server.app"


def test_app_routes_registered():
    from tool_evolution.server.app import app

    # 路由注册的稳定契约是 OpenAPI schema：FastAPI 0.11x 会把 include_router 的路由
    # 摊平进 app.routes，而 0.16x 起改为惰性 _IncludedRouter 包装（app.routes 里只剩
    # 顶层 Route）。两种实现下 app.openapi()["paths"] 都包含全部已注册路由，所以按
    # OpenAPI paths 断言，既不依赖 FastAPI 内部表示，也不放松“路由必须已注册”的验收。
    paths = set((app.openapi() or {}).get("paths") or {})
    for expected in ("/health", "/api/traces/report", "/api/skills/discoveries",
                     "/api/rules", "/api/analytics/summary", "/api/canary/{skill_id}/promote",
                     "/api/memory/search"):
        assert expected in paths, f"route {expected} is not registered (openapi paths: {sorted(paths)})"

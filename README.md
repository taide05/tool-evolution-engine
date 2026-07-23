# Tool Evolution Engine

Agent tool-calling adaptive optimization engine -- enables Agent tool-calling capability to continuously self-improve.

## Architecture

Four-layer pipeline: Collection -> Analysis -> Knowledge -> Governance

```
Traces (Collection)
    |
    v
Analysis (Classifier / KDE / DAG Miner / Distiller)
    |
    v
Knowledge (Rule Engine / Param Templates / Skill Packs)
    |
    v
Governance (Credit Scoring / Canary Promotion / A/B Rollback / MCP Bridge)
```

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/seed_demo_data.py
python scripts/run_demo.py
uvicorn tool_evolution.server.app:app --reload
```

## Core Modules

| Module | Function | File |
|--------|----------|------|
| Trace Collection | Full-chain tool call snapshots + FTS5 full-text index | `collection/tracer.py` |
| Failure Classification | RandomForest 5-class classification + counterfactual rule distillation | `analysis/classifier.py` |
| KDE Analysis | Parameter distribution fitting -> auto defaults + bounds | `analysis/kde_analyzer.py` |
| DAG Mining | Frequent subgraph -> composite skill pack auto-discovery | `analysis/dag_miner.py` |
| Skill Governance | 3D credit scoring + canary rollout + A/B rollback | `governance/governor.py` |
| MCP Bridge | Memory system bidirectional data flow + SQLite fallback | `governance/mcp_bridge.py` |

## API Endpoints

- `POST /api/traces/report` -- Report trace
- `POST /api/traces/seed` -- Seed data injection
- `GET /api/skills/discoveries` -- List discovered skills
- `GET /api/skills/deployed` -- List deployed skills
- `GET /api/rules` -- List rules
- `GET /api/analytics/summary` -- Analysis summary
- `POST /api/canary/{id}/promote` -- Promote skill
- `POST /api/canary/{id}/compare` -- A/B comparison

## Project Layout

```
tool-evolution-engine/
  src/tool_evolution/
    collection/     # Trace collection (schemas, store, tracer)
    analysis/       # Analysis (classifier, distiller, kde_analyzer, dag_miner)
    knowledge/      # Knowledge (rule_engine, param_template, skill_pack)
    governance/     # Governance (governor, mcp_bridge)
    utils/          # Utilities (config, database)
  scripts/
    seed_demo_data.py  # Generate 200 simulated traces
    run_demo.py        # End-to-end 7-step demo pipeline
  tests/            # Test suite
```

## License

MIT

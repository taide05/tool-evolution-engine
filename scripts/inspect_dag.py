import asyncio, json, aiosqlite
async def main():
    conn = await aiosqlite.connect("data/engine.db")
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT name, dag_definition, frequency FROM discovered_skills")
    rows = await cursor.fetchall()
    for r in rows:
        d = json.loads(r["dag_definition"])
        edges = [(e["from"], e["to"]) for e in d.get("edges", [])]
        nodes = [n["tool_name"] for n in d.get("nodes", [])]
        print(f"DB name: {r['name']}")
        print(f"  Nodes JSON: {nodes}")
        print(f"  Edges JSON: {edges}")
        print()
    await conn.close()
asyncio.run(main())

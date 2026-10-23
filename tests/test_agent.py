"""End-to-end check of the forecasting agent over stdio: list tools, inspect, forecast, scenarios.

Run: python tests/test_agent.py [--zeroshot]
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CSV = str(Path(__file__).resolve().parents[1] / "examples" / "example_well.csv")
MODEL = "zeroshot" if "--zeroshot" in sys.argv else "adapter"


async def main():
    env = {k: v for k, v in os.environ.items() if k.startswith(("GWFM_", "HF_")) or k in ("HOME", "PATH")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "gwfm.mcp_server"], env=env)
    out = tempfile.mkdtemp(prefix="gwfm_test_")
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = await s.list_tools()
        print("tools:", [(t.name, t.annotations.read_only_hint, t.annotations.destructive_hint) for t in tools.tools])
        res = await s.call_tool("inspect_series", {"csv_path": CSV})
        print(res.content[0].text[:300])
        res = await s.call_tool("forecast_groundwater", {"csv_path": CSV, "horizon_days": 14, "model": MODEL,
                                                         "future_rain": [0, 0, 30, 60, 10] + [0] * 9, "out_dir": out})
        assert not res.content[0].text.startswith("Error"), res.content[0].text
        j = json.loads(res.content[0].text)
        print(j["mode"], j["summary"], j["figure"])
        res = await s.call_tool("compare_scenarios", {"csv_path": CSV, "horizon_days": 30, "model": MODEL, "out_dir": out,
                                                      "scenarios": [{"name": "wet", "rain": [20.0] * 30}]})
        j = json.loads(res.content[0].text)
        print({k: (v["total_rain_mm"], v["end_median"]) for k, v in j["scenarios"].items()})
        res = await s.call_tool("model_info", {})
        print(res.content[0].text[:120])


asyncio.run(main())

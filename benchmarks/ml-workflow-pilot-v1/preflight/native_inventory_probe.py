"""Zero-turn native app-server inventory probe; never calls turn/start."""

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

REPO = Path.cwd()
OUT = REPO / "out/ml-workflow-pilot-preflight"
PYTHON = "/Users/jasonling/.pyenv/versions/3.11.9/bin/python"


async def probe(arm):
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"ml-workflow-preflight-{arm}-")).resolve()
    cfg = {
        f"mcp_servers.{name}.enabled": False
        for name in (
            "computer-use",
            "hf-mcp-server",
            "huggingface",
            "node_repl",
            "runpod",
            "arxiv-research",
            "ml-experiments",
        )
    }
    for name in (
        "computer-use",
        "hf-mcp-server",
        "huggingface",
        "node_repl",
        "runpod",
        "arxiv-research",
        "ml-experiments",
    ):
        if name in ("hf-mcp-server", "runpod"):
            cfg[f"mcp_servers.{name}.url"] = (
                "https://huggingface.co/mcp?login"
                if name == "hf-mcp-server"
                else "https://mcp.getrunpod.io/"
            )
        else:
            cfg[f"mcp_servers.{name}.command"] = "/usr/bin/false"
    cfg.update(
        {
            "features.plugins": False,
            "features.apps": False,
            "features.multi_agent": False,
            "features.memories": False,
            "web_search": "live",
            "approval_policy": "never",
            "model": "gpt-6.1-sol",
            "model_reasoning_effort": "medium",
            "default_permissions": "ml-pilot",
            "permissions.ml-pilot.filesystem": {
                ":minimal": "read",
                ":workspace_roots": "write",
                "/Users/jasonling/.local/bin": "read",
                "/Users/jasonling/.codex/packages/standalone": "read",
                str(Path("/Users/jasonling/.local/bin/codex").resolve()): "read",
            },
            "permissions.ml-pilot.network.enabled": False,
        }
    )
    if arm == "B":
        roots = OUT / arm
        servers = {
            "arxiv-research": (
                "arxiv_research_mcp.py",
                ["--tool-profile", "sources", "--library-root", str(roots / "library")],
            ),
            "ml-experiments": (
                "ml_experiment_mcp.py",
                [
                    "--store-root",
                    str(roots / "records"),
                    "--artifact-root",
                    str(roots / "artifacts"),
                    "--library-root",
                    str(roots / "library"),
                ],
            ),
        }
        for name, (script, args) in servers.items():
            for key, value in {
                "command": PYTHON,
                "args": [str(REPO / "scripts" / script), *args],
                "cwd": str(REPO),
                "enabled": True,
                "startup_timeout_sec": 30,
            }.items():
                cfg[f"mcp_servers.{name}.{key}"] = value
        (roots / "artifacts").mkdir(exist_ok=True)
    command = ["codex", "app-server"]
    for key, value in cfg.items():
        encoded = (
            "{" + ", ".join(json.dumps(k) + "=" + json.dumps(v) for k, v in value.items()) + "}"
            if isinstance(value, dict)
            else json.dumps(value)
        )
        command.extend(["-c", key + "=" + encoded])
    stderr = (OUT / arm / "stderr.log").open("w")
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=work,
        env={**os.environ, "PWD": str(work)},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr,
        start_new_session=True,
    )
    calls, events = [], []

    async def request(method, params):
        ident = len(calls) + 1
        calls.append(method)
        proc.stdin.write(
            (json.dumps({"id": ident, "method": method, "params": params}) + "\n").encode()
        )
        await proc.stdin.drain()
        while True:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=60)
            if not raw:
                raise RuntimeError("native app-server closed stdout")
            item = json.loads(raw)
            if item.get("id") == ident:
                if "error" in item:
                    raise RuntimeError(json.dumps(item["error"]))
                return item["result"]
            if "id" in item and "method" in item:
                # No approval/interaction is expected in a zero-turn inventory.
                raise RuntimeError("unexpected server request: " + item["method"])
            events.append(item.get("method"))

    result = {"arm": arm, "probe": "zero_turn_inventory", "model_turns": 0, "config_overrides": cfg}
    try:
        result["initialize"] = await request(
            "initialize",
            {
                "clientInfo": {"name": "ml_workflow_inventory", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        proc.stdin.write(b'{"method":"initialized"}\n')
        await proc.stdin.drain()
        config = await request("config/read", {"includeLayers": False, "cwd": str(work)})
        safe = (
            "model",
            "model_provider",
            "model_reasoning_effort",
            "web_search",
            "sandbox_mode",
            "approval_policy",
            "features",
        )
        result["effective_config"] = {key: config["config"].get(key) for key in safe}
        result["configured_mcp_enabled"] = {
            name: value.get("enabled", True)
            for name, value in config["config"].get("mcp_servers", {}).items()
        }
        thread = await request(
            "thread/start",
            {
                "cwd": str(work),
                "ephemeral": True,
                "permissions": "ml-pilot",
                "approvalPolicy": "never",
                "allowProviderModelFallback": False,
            },
        )
        result["thread"] = {
            key: thread.get(key)
            for key in (
                "model",
                "modelProvider",
                "reasoningEffort",
                "sandbox",
                "activePermissionProfile",
                "instructionSources",
                "runtimeWorkspaceRoots",
            )
        }
        inventory, cursor = [], None
        while True:
            page = await request(
                "mcpServerStatus/list",
                {
                    "threadId": thread["thread"]["id"],
                    "detail": "full",
                    "cursor": cursor,
                    "limit": 100,
                },
            )
            inventory.extend(page["data"])
            cursor = page.get("nextCursor")
            if not cursor:
                break
        result["mcp_inventory"] = inventory
        result["status"] = "inventory_observed"
        inside = work / "canary.txt"
        inside.write_text("public probe canary\n")
        outside = OUT / "not-for-child.txt"
        outside.write_text("outside-workspace probe canary\n")
        result["filesystem_checks"] = {}
        for label, path in [("workspace", inside), ("outside_workspace", outside)]:
            response = await request(
                "command/exec",
                {
                    "command": ["/bin/cat", str(path)],
                    "cwd": str(work),
                    "permissionProfile": "ml-pilot",
                    "timeoutMs": 3000,
                    "outputBytesCap": 1024,
                },
            )
            result["filesystem_checks"][label] = response
        assert result["filesystem_checks"]["workspace"]["exitCode"] == 0
        assert result["filesystem_checks"]["outside_workspace"]["exitCode"] != 0
    except Exception as exc:
        result.update(status="preflight_error", error=str(exc))
    finally:
        result.update(methods_requested=calls, notification_methods=sorted(set(events)))
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        stderr.close()
        (OUT / arm / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arm": arm,
                "status": result["status"],
                "error": result.get("error"),
                "model": result.get("thread", {}).get("model"),
                "servers": [
                    (x.get("name"), len(x.get("tools", {})))
                    for x in result.get("mcp_inventory", [])
                ],
            }
        ),
        flush=True,
    )


async def main():
    for arm in ("A", "B"):
        await probe(arm)


asyncio.run(main())

"""Frozen paired reader evaluation. No optimizer, reward admission, or cloud provisioning.

Freeze on the training machine, then copy the session to the Docker controller.
Only an explicit later command opens reference contents or generates model outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata as metadata
import json
import math
import secrets
import shutil
import signal
import statistics
import time
from pathlib import Path

import httpx

from src.eval.passage_pilot_replay import PROMPT, PROMPT_SHA, frozen_runtime
from src.eval.reader_comparison_run import action_metrics
from src.eval.reader_improvement import digest, file_hash, inside, load
from src.research.agent import load_snapshot
from src.training.prime_budget import loopback_engine, proxy
from src.training.prime_run import BASE_REVISION, RENDERER_REVISION, V5_SHA, validate_plan, write
from src.training.reader_data import ROOT, load_bundle

PROTOCOL = ROOT / "examples/reader-rl-pilot-v1/protocol.json"
BUNDLE = ROOT / "data/rl/reader-facts-v1"
ARMS = ("unchanged_v5", "final_rl_step_10")
IMPLEMENTATION = (
    "src/training/reader_evaluation.py",
    "src/training/verifiers_adapter.py",
    "src/training/prime_budget.py",
    "src/training/reader_rollout.py",
    "src/training/reader_data.py",
    "src/eval/reader_comparison_run.py",
    "examples/verifiers-port/prime-base-files.json",
    "examples/verifiers-port/prime-rl-smoke/native-source-proof.json",
)


def freeze(training, output):
    """Require a completed, originally initialized ten-update run before unlocking."""
    training, output = Path(training).resolve(), Path(output).resolve()
    plan = validate_plan(training)
    protocol_path = inside(ROOT, plan.get("pilot_protocol", PROTOCOL))
    protocol = load(protocol_path)
    phase = protocol["phases"]["training"]
    expected_limits = dict(
        updates=10,
        requests=phase["max_generation_requests"],
        reserved_tokens=phase["max_reserved_tokens"],
        seconds=phase["max_work_seconds"],
    )
    result = load(training / "result.json")
    if (
        plan["limits"] != expected_limits
        or plan["v5_sha256"] != V5_SHA
        or set(plan["train_case_ids"]) != set(protocol["training_case_ids"])
        or result.get("verified_updates") != 10
        or result.get("attempted_update_steps") != list(range(1, 11))
        or result.get("unverified_attempts_may_have_updated_weights") is not False
        or result.get("all_recorded_containers_absent") is not True
    ):
        raise ValueError("only the complete predeclared ten-update pilot can unlock evaluation")
    previous_sha, previous_optimizer = V5_SHA, None
    for step in range(1, 11):
        verified = load(training / f"cycle-{step}/verified.json")
        loaded = load(training / f"cycle-{step}/loaded.json")
        prepared = load(training / f"cycle-{step}/prepared.json")
        scorer = load(training / f"cycle-{step}/scorer.json")
        if (
            verified.get("step") != step
            or verified.get("optimizer_steps") != [step]
            or verified.get("reload_matches_export") is not True
            or verified.get("startup_matches_sampled_policy") is not True
            or verified.get("adapter_sha256")
            != file_hash(training / f"cycle-{step}/adapter/adapter_model.safetensors")
        ):
            raise ValueError("missing verified checkpoint continuity")
        if (loaded.get("version") != step - 1 or loaded.get("sha256") != previous_sha
                or loaded.get("load_rpc_completed") is not True
                or loaded.get("engine_reported_path_matches") is not True
                or prepared.get("version") != step - 1
                or prepared.get("scorer_passed") is not True
                or prepared.get("batch_sha256") != file_hash(
                    training / f"cycle-{step}/microbatches.bin")
                or (step > 1 and prepared.get("optimizer_sha256") != previous_optimizer)
                or type(scorer.get("tokens")) is not int or scorer["tokens"] <= 0
                or any(type(scorer.get(k)) not in {int, float}
                       or not math.isfinite(scorer[k]) or not 0 <= scorer[k] <= limit
                       for k, limit in (("mean", .03), ("max", .25)))):
            raise ValueError("missing sampled-policy, optimizer or native scoring proof")
        previous_sha, previous_optimizer = verified["adapter_sha256"], verified["optimizer_sha256"]
    candidate = training / "cycle-10/adapter"
    if file_hash(candidate / "adapter_model.safetensors") == V5_SHA:
        raise ValueError("candidate is unchanged V5")
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "training-evidence"
    evidence.mkdir()
    for name in ["plan.json", "started.json", "result.json"] + [
        f"cycle-{n}/{receipt}.json"
        for n in range(1, 11)
        for receipt in ("verified", "scorer", "loaded", "prepared")
    ]:
        target = evidence / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(training / name, target)
    for arm, source in zip(ARMS, (Path(plan["assets"]) / "v5", candidate), strict=True):
        target = output / arm
        target.mkdir()
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copyfile(source / name, target / name)
    files = {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}
    session = {
        "schema": "reader-paired-evaluation-v1",
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_hash(protocol_path),
        "bundle_sha256": file_hash(BUNDLE / "manifest.json"),
        "prompt_sha256": PROMPT_SHA,
        "frozen_runtime": frozen_runtime(),
        "artifacts": files,
        "implementation_sha256": {n: file_hash(ROOT / n) for n in IMPLEMENTATION},
        "question_ids": protocol["evaluation_question_ids"],
        "arms": {a: files[a + "/adapter_model.safetensors"] for a in ARMS},
        "training_eligible": False,
        "independent_human_review": None,
    }
    write(output / "session.json", session)
    validate_session(output)
    return session


def validate_session(output):
    output = Path(output).resolve()
    session = load(output / "session.json")
    protocol_path = inside(ROOT, session.get("protocol_path", PROTOCOL))
    protocol = load(protocol_path)
    if (
        session.get("schema") != "reader-paired-evaluation-v1"
        or session.get("protocol_sha256") != file_hash(protocol_path)
        or session.get("bundle_sha256") != file_hash(BUNDLE / "manifest.json")
        or session.get("prompt_sha256") != PROMPT_SHA
        or session.get("frozen_runtime") != frozen_runtime()
        or session.get("question_ids") != protocol["evaluation_question_ids"]
        or session.get("training_eligible") is not False
        or session.get("arms", {}).get(ARMS[0]) != V5_SHA
        or session.get("arms", {}).get(ARMS[1]) in {None, V5_SHA}
        or set(session.get("arms", {})) != set(ARMS)
        or set(session.get("implementation_sha256", {})) != set(IMPLEMENTATION)
    ):
        raise ValueError("evaluation session identity drift")
    required = {f"{a}/{n}" for a in ARMS
                for n in ("adapter_model.safetensors", "adapter_config.json")}
    required |= {"training-evidence/" + n for n in ("plan.json", "started.json", "result.json")}
    required |= {f"training-evidence/cycle-{n}/{receipt}.json" for n in range(1, 11)
                 for receipt in ("verified", "scorer", "loaded", "prepared")}
    if set(session["artifacts"]) != required:
        raise ValueError("evaluation lacks the complete frozen training/adapter evidence")
    for name, expected in session["artifacts"].items():
        if file_hash(inside(output, name)) != expected:
            raise ValueError("frozen evaluation artifact changed: " + name)
    training = load(output / "training-evidence/plan.json")
    if (inside(ROOT, training.get("pilot_protocol", PROTOCOL)) != protocol_path
            or (training.get("pilot_protocol") is not None
                and training.get("pilot_protocol_sha256") != file_hash(protocol_path))):
        raise ValueError("evaluation protocol differs from the frozen training protocol")
    revision = protocol["identities"].get("reward_revision")
    if (revision is not None and file_hash(inside(ROOT, revision)) !=
            protocol["identities"].get("reward_revision_sha256")):
        raise ValueError("frozen training reward revision changed")
    for arm in ARMS:
        if file_hash(output / arm / "adapter_model.safetensors") != session["arms"][arm]:
            raise ValueError("evaluation adapter changed")
    for name, expected in session["implementation_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("evaluation implementation changed: " + name)
    _, roles, _, _ = load_bundle(BUNDLE)  # Locked reference bytes are hashed, not decoded.
    cases = roles["model_evaluation_locked"]
    if [c["question_id"] for c in cases] != session["question_ids"]:
        raise ValueError("evaluation case order drift")
    return session, cases


def session_protocol(output):
    """Use the protocol frozen with this session, after validate_session's checks."""
    session = load(Path(output) / "session.json")
    return load(inside(ROOT, session.get("protocol_path", PROTOCOL)))


def source_windows(case):
    _, docs = load_snapshot(inside(BUNDLE, case["snapshot"]))
    doc = docs[case["doc_id"]]
    return [
        {**w, "doc_id": case["doc_id"], "quote": doc["text"][w["start"] : w["end"]]}
        for w in case["scope"]["windows"]
    ]


def open_references(output):
    """Only explicit post-candidate export decodes the twelve locked references."""
    output = Path(output)
    _, cases = validate_session(output)
    if (output / "started.json").exists():
        raise ValueError("reference review must precede model outcomes")
    pack = output / "reference-pack.json"
    if pack.exists():
        raise FileExistsError("reference pack already exported; do not replace the review")
    rows = [
        {
            "question_id": c["question_id"],
            "question": c["question"],
            "scope": c["scope"],
            "source_windows": source_windows(c),
            "reference": load(BUNDLE / "model-evaluation-locked" / (c["case_id"] + ".json")),
            "in_scope_answerable": None,
            "reference_unambiguous": None,
            "notes": None,
        }
        for c in cases
    ]
    write(
        pack,
        {
            "session_sha256": file_hash(output / "session.json"),
            "reviewer_kind": None,
            "reviewer_identity": None,
            "reviewed_at": None,
            "independent_human_review": None,
            "rows": rows,
        },
    )


def reference_review(output, cases):
    """Validate attribution and unchanged source/reference bindings, not semantics."""
    review = load(output / "reference-pack.json")
    if (
        review.get("session_sha256") != file_hash(output / "session.json")
        or review.get("reviewer_kind") not in {"human", "model_assisted"}
        or not review.get("reviewer_identity")
        or not review.get("reviewed_at")
        or (
            review["reviewer_kind"] == "model_assisted"
            and review.get("independent_human_review") is not None
        )
        or len(review.get("rows", [])) != len(cases)
    ):
        raise ValueError("complete attributed reference review required before inference")
    for row, case in zip(review["rows"], cases, strict=True):
        reference = BUNDLE / "model-evaluation-locked" / (case["case_id"] + ".json")
        if (
            row.get("question_id") != case["question_id"]
            or row.get("question") != case["question"]
            or row.get("scope") != case["scope"]
            or row.get("source_windows") != source_windows(case)
            or digest(row.get("reference")) != digest(load(reference))
            or row.get("in_scope_answerable") is not True
            or row.get("reference_unambiguous") is not True
            or not row.get("notes")
        ):
            raise ValueError("unresolved/stale reference review; evaluation remains inconclusive")
    return file_hash(output / "reference-pack.json")


def evaluation_case(output, data):
    session, cases = validate_session(output)
    if (
        data.evaluation_session_sha256 != file_hash(Path(output) / "session.json")
        or data.manifest_sha256 != session["bundle_sha256"]
        or Path(data.bundle).resolve() != BUNDLE.resolve()
    ):
        raise ValueError("evaluation task is not bound to the frozen session")
    case = next((c for c in cases if c["case_id"] == data.case_id), None)
    if (
        case is None
        or data.id != case["question_id"]
        or data.prompt != case["question"]
        or data.system_prompt != PROMPT
    ):
        raise ValueError("only the unchanged locked question and prompt are allowed")
    return BUNDLE, case


def check_tokenizer(tokenizer):
    catalog = load(ROOT / "examples/verifiers-port/prime-base-files.json")
    for name, expected in catalog["files"].items():
        if name not in {"config.json", "merges.txt", "tokenizer.json",
                        "tokenizer_config.json", "vocab.json"}:
            continue
        raw = (tokenizer / name).read_bytes()
        actual = (
            hashlib.sha256(raw).hexdigest()
            if expected["lfs_sha256"]
            else hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        )
        if actual != (expected["lfs_sha256"] or expected["git_blob_id"]):
            raise ValueError("local tokenizer differs from pinned Hub revision: " + name)


def check_native_sources():
    import prime_rl.orchestrator.clients as native_clients

    renderer = json.loads(metadata.distribution("renderers").read_text("direct_url.json") or "{}")
    if renderer.get("vcs_info", {}).get("commit_id") != RENDERER_REVISION:
        raise ValueError("evaluation requires the pinned renderer commit")
    proof = load(ROOT / "examples/verifiers-port/prime-rl-smoke/native-source-proof.json")
    for name, sha in proof["matching_source_files"].items():
        package = Path(native_clients.__file__).parents[1]
        if file_hash(package / name.removeprefix("src/prime_rl/")) != sha:
            raise ValueError("evaluation native source drift: " + name)


async def collect_episode(output, case, tokenizer, url, arm, session, seed):
    """Native non-thinking renderer/TrainClient, with no reward function or TrainSink."""
    import verifiers.v1 as vf
    from renderers.configs import DefaultRendererConfig

    from src.training.verifiers_adapter import (
        ReaderEnvConfig,
        ReaderState,
        ReaderTaskData,
        ReaderTasksetConfig,
        local_agent_config,
    )
    check_native_sources()

    class EvaluationTask(vf.Task[ReaderTaskData, ReaderState]):
        @property
        def key(self):
            return self.data.case_id

    task = EvaluationTask(
        ReaderTaskData(
            id=case["question_id"],
            case_id=case["case_id"],
            bundle=BUNDLE,
            manifest_sha256=session["bundle_sha256"],
            prompt=case["question"],
            system_prompt=PROMPT,
            evaluation_session_sha256=file_hash(output / "session.json"),
        )
    )
    config = local_agent_config(
        url,
        output / arm / "rollouts",
        model_identity={
            "backend": "native_prime_vllm",
            "arm": arm,
            "adapter_sha256": session["arms"][arm],
            "weights_verified": False,
            "training_eligible": False,
        },
    )
    config.harness.token_contract_only = True
    config.harness.evaluation_session = output
    config.client = vf.TrainClientConfig(
        base_url=url,
        api_key_var="ENVOY_VERIFIERS_UNUSED",
        renderer_model_name=str(tokenizer),
        renderer=DefaultRendererConfig(enable_thinking=False),
    )
    config.sampling = vf.SamplingConfig(
        **config.sampling.model_dump(), top_k=-1, min_p=0.0, seed=seed
    )
    env = vf.load_environment(ReaderEnvConfig(taskset=ReaderTasksetConfig(), agent=config))
    async with env.serving():
        episode = await env.run_episode(
            task, vf.ModelContext(model="envoy", client=config.client, sampling=config.sampling)
        )
    if any(t.agent.trainable or t.rewards for t in episode.traces):
        raise ValueError("evaluation traces must remain untrainable and unscored")
    return episode


def model_root(engine, expected):
    """Check alias/path, without claiming tensor-level engine attestation."""
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
        with client.stream("GET", loopback_engine(engine) + "/v1/models") as response:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > 1024 * 1024:
                    raise ValueError("model metadata exceeds 1 MiB")
    models = json.loads(raw)["data"]
    if not any(m.get("id") == "envoy" and m.get("root") == expected for m in models):
        raise ValueError("engine alias does not report the frozen arm's adapter root")


async def run_arm(output, arm, engine, tokenizer, backend_receipt):
    """Execute once per arm. External native loader supplies a hash-bound receipt.

    The same session is portable between hosts; backend paths are declarations,
    checked against /v1/models, not evidence of tensor-level engine attestation.
    """
    output, tokenizer = Path(output).resolve(), Path(tokenizer).resolve()
    session, cases = validate_session(output)
    ref_sha = reference_review(output, cases)
    if arm not in ARMS or tokenizer.name != BASE_REVISION or not tokenizer.is_dir():
        raise ValueError("unknown arm or unpinned local tokenizer")
    check_tokenizer(tokenizer)
    backend = load(backend_receipt)
    if (
        backend.get("session_sha256") != file_hash(output / "session.json")
        or backend.get("arm") != arm
        or backend.get("adapter_sha256") != session["arms"][arm]
        or backend.get("load_rpc_completed") is not True
        or not backend.get("gpu")
        or not backend.get("versions")
        or not backend.get("adapter_root", "").startswith("/")
    ):
        raise ValueError("explicit arm-specific native load/hardware receipt required")
    if arm == ARMS[1] and not (output / ARMS[0] / "complete.json").exists():
        raise ValueError("V5 must complete before the RL arm")
    if arm == ARMS[1]:
        baseline = load(output / ARMS[0] / "complete.json")["backend"]
        if any(backend[k] != baseline[k] for k in ("gpu", "versions")):
            raise ValueError("paired arms require comparable hardware and package versions")
    started = output / "started.json"
    if not started.exists():
        phase = session_protocol(output)["phases"]["paired_evaluation"]
        with started.open("x") as stream:
            json.dump(
                {
                    "deadline": time.time() + phase["max_work_seconds"],
                    "session_sha256": file_hash(output / "session.json"),
                    "reference_review_sha256": ref_sha,
                },
                stream,
            )
    state = load(started)
    if (
        state["session_sha256"] != file_hash(output / "session.json")
        or state["reference_review_sha256"] != ref_sha
    ):
        raise ValueError("session/reference review changed after starting")
    # Never overwrite or retry a partial arm, including a failure before dispatch.
    with (output / arm / "attempted.json").open("x") as stream:
        json.dump({"backend": backend, "started": time.time()}, stream)
    model_root(engine, backend["adapter_root"])
    phase = session_protocol(output)["phases"]["paired_evaluation"]
    plan = {
        "limits": {
            "requests": phase["max_generation_requests"],
            "reserved_tokens": phase["max_reserved_tokens"],
        }
    }
    with proxy(output, plan, backend, engine=engine) as url:
        for index, case in enumerate(cases):
            model_root(engine, backend["adapter_root"])
            ledger = (
                load(output / "dispatches.json")
                if (output / "dispatches.json").exists()
                else {"requests": [], "reserved_tokens": 0}
            )
            before, requests_before, start = (
                ledger["reserved_tokens"],
                len(ledger["requests"]),
                time.monotonic(),
            )
            episode = await collect_episode(output, case, tokenizer, url, arm, session, 42 + index)
            elapsed = time.monotonic() - start
            record = episode.to_record(float_decimals=None)
            # VF excludes task state from its JSON record. Preserve the Docker
            # packet explicitly so a portable review never relies on host paths.
            record["docker_episode"] = (
                episode.traces[0].state.episode if len(episode.traces) == 1 else None
            )
            write(output / arm / f"episode-{index:02}.json", record)
            ledger = load(output / "dispatches.json")
            write(
                output / arm / f"measure-{index:02}.json",
                {
                    "question_id": case["question_id"],
                    "seconds": elapsed,
                    "reserved_tokens": ledger["reserved_tokens"] - before,
                    "requests": len(ledger["requests"]) - requests_before,
                    "seed": 42 + index,
                    "episode_sha256": file_hash(output / arm / f"episode-{index:02}.json"),
                },
            )
            if not episode.ok or len(episode.traces) != 1 or not episode.traces[0].ok:
                raise ValueError("unhealthy evaluation: preserve outputs, stop, no replacement")
    write(
        output / arm / "complete.json",
        {
            "episodes": len(cases),
            "backend": backend,
            "session_sha256": file_hash(output / "session.json"),
        },
    )


def export_review(output):
    """Derived identity-masked review pack; raw records and private mapping are retained."""
    output = Path(output)
    _, cases = validate_session(output)
    reference_review(output, cases)
    target = output / "output-review.json"
    if target.exists():
        raise FileExistsError("review pack already exists")
    rows, mapping = [], {}
    for arm in ARMS:
        if not (output / arm / "complete.json").exists():
            raise ValueError("all 24 episodes required; partial runs remain inconclusive")
        for i, case in enumerate(cases):
            path = output / arm / f"episode-{i:02}.json"
            episode = load(path)
            packet = episode["docker_episode"]["packet"]
            code = secrets.token_hex(12)
            mapping[code] = {"arm": arm, "index": i, "episode_sha256": file_hash(path)}
            rows.append(
                {
                    "review_id": code,
                    "question_id": case["question_id"],
                    "question": case["question"],
                    "source_windows": source_windows(case),
                    "candidate_claims": packet["candidate_claims"],
                    "status": packet["status"],
                    "evidence": packet["evidence"],
                    "inspected_passage_windows": packet["inspected_passage_windows"],
                    "complete_correct_supported": None,
                    "false_refusal": None,
                    "material_claims": [],
                    "claim_inventory_complete": None,
                    "requested_fact_review": None,
                    "unresolved": None,
                    "notes": None,
                }
            )
    secrets.SystemRandom().shuffle(rows)
    write(output / "private-arm-map.json", mapping)
    write(
        target,
        {
            "session_sha256": file_hash(output / "session.json"),
            "reviewer_kind": None,
            "reviewer_identity": None,
            "reviewed_at": None,
            "independent_human_review": None,
            "rows": rows,
        },
    )
    write(
        output / "output-review-bindings.json",
        {
            "private_map_sha256": file_hash(output / "private-arm-map.json"),
            "telemetry_sha256": {
                str(p.relative_to(output)): file_hash(p)
                for pattern in ("*/episode-*.json", "*/measure-*.json", "*/complete.json",
                                "*/attempted.json", "dispatches.json", "started.json",
                                "engine-request-*.json", "engine-response-*.json")
                for p in output.glob(pattern)
            },
            "rows": {
                r["review_id"]: digest(
                    {
                        k: v
                        for k, v in r.items()
                        if k
                        not in {
                            "complete_correct_supported",
                            "claim_inventory_complete",
                            "requested_fact_review",
                            "false_refusal",
                            "material_claims",
                            "unresolved",
                            "notes",
                        }
                    }
                )
                for r in rows
            },
        },
    )


def summarize(output):
    """Apply the predeclared engineering gate to attributed claim-level judgments."""
    output = Path(output)
    _, cases = validate_session(output)
    reference_review(output, cases)
    review, mapping = load(output / "output-review.json"), load(output / "private-arm-map.json")
    bindings = load(output / "output-review-bindings.json")
    for name, sha in bindings["telemetry_sha256"].items():
        if file_hash(inside(output, name)) != sha:
            raise ValueError("raw episode or telemetry changed after review export")
    if (
        bindings["private_map_sha256"] != file_hash(output / "private-arm-map.json")
        or review.get("session_sha256") != file_hash(output / "session.json")
        or review.get("reviewer_kind") not in {"human", "model_assisted"}
        or not review.get("reviewer_identity")
        or not review.get("reviewed_at")
        or (
            review["reviewer_kind"] == "model_assisted"
            and review.get("independent_human_review") is not None
        )
        or len(review.get("rows", [])) != len(cases) * 2
        or {r["review_id"] for r in review["rows"]} != set(mapping)
    ):
        raise ValueError("complete source-bound attributed paired review required")
    cells = {a: {} for a in ARMS}
    for row in review["rows"]:
        key = row["review_id"]
        source = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "complete_correct_supported",
                "claim_inventory_complete",
                "requested_fact_review",
                "false_refusal",
                "material_claims",
                "unresolved",
                "notes",
            }
        }
        if (
            digest(source) != bindings["rows"][key]
            or any(
                type(row.get(k)) is not bool
                for k in ("complete_correct_supported", "false_refusal", "unresolved")
            )
            or row["unresolved"]
            or not row.get("notes")
        ):
            raise ValueError("missing, uncertain or altered source-visible review")
        if (row.get("claim_inventory_complete") is not True
                or not row.get("requested_fact_review")):
            raise ValueError("review all material claims and requested facts explicitly")
        claims = row["material_claims"]
        if row["candidate_claims"] and not row["false_refusal"] and not claims:
            raise ValueError("asserted answers require a material-claim inventory")
        if not isinstance(claims, list) or any(
            not c.get("claim")
            or type(c.get("supported")) is not bool
            or not c.get("notes")
            or not isinstance(c.get("evidence_ids"), list)
            or not set(c["evidence_ids"]) <= {e["evidence_id"] for e in row["evidence"]}
            or (c["supported"] and not c["evidence_ids"])
            for c in claims
        ):
            raise ValueError("each material claim needs a cited-source support judgment")
        unsupported = any(not c["supported"] for c in claims)
        if row["complete_correct_supported"] and (
            unsupported or row["false_refusal"] or not claims or row["status"] != "evidence_found"
        ):
            raise ValueError("supported answer conflicts with its claim/refusal judgments")
        arm, index = mapping[key]["arm"], mapping[key]["index"]
        path = output / arm / f"episode-{index:02}.json"
        measure = load(output / arm / f"measure-{index:02}.json")
        if (
            file_hash(path) != mapping[key]["episode_sha256"]
            or measure["episode_sha256"] != file_hash(path)
            or row["question_id"] != cases[index]["question_id"]
            or measure["question_id"] != row["question_id"]
            or measure["seed"] != 42 + index
            or type(measure["reserved_tokens"]) is not int
            or measure["reserved_tokens"] <= 0
            or type(measure["seconds"]) not in {int, float}
            or not math.isfinite(measure["seconds"])
            or measure["seconds"] <= 0
        ):
            raise ValueError("missing or incomparable episode telemetry")
        packet = load(path)["docker_episode"]["packet"]
        cells[arm][index] = {
            "supported": row["complete_correct_supported"],
            "unsupported": unsupported,
            "false_refusal": row["false_refusal"],
            "execution_failure": bool(action_metrics(packet)["execution_error_steps"]),
            "reserved_tokens": measure["reserved_tokens"],
            "seconds": measure["seconds"],
        }
    if any(set(cells[a]) != set(range(len(cases))) for a in ARMS):
        raise ValueError("duplicate/missing paired questions")
    counts = {
        a: {
            k: sum(c[k] for c in rows.values())
            for k in (
                "supported",
                "unsupported",
                "false_refusal",
                "execution_failure",
                "reserved_tokens",
            )
        }
        for a, rows in cells.items()
    }
    pairs = [
        {"question_id": c["question_id"], "v5": cells[ARMS[0]][i], "rl": cells[ARMS[1]][i]}
        for i, c in enumerate(cases)
    ]
    wins = sum(p["rl"]["supported"] and not p["v5"]["supported"] for p in pairs)
    losses = sum(p["v5"]["supported"] and not p["rl"]["supported"] for p in pairs)
    newly_unsupported = sum(p["rl"]["unsupported"] and not p["v5"]["unsupported"] for p in pairs)
    ratios = {
        "mean_reserved_token": counts[ARMS[1]]["reserved_tokens"]
        / counts[ARMS[0]]["reserved_tokens"]
    }
    for label, fn in (
        ("median_episode_latency", statistics.median),
        ("p95_episode_latency", lambda v: sorted(v)[math.ceil(0.95 * len(v)) - 1]),
    ):
        ratios[label] = fn([c["seconds"] for c in cells[ARMS[1]].values()]) / fn(
            [c["seconds"] for c in cells[ARMS[0]].values()]
        )
    rule = session_protocol(output)["success_rule"]
    positive = (
        wins - losses >= rule["minimum_net_supported_answer_gain"]
        and losses <= rule["maximum_supported_answer_regressions"]
        and newly_unsupported <= rule["maximum_new_unsupported_questions"]
        and all(
            counts[ARMS[1]][k] <= counts[ARMS[0]][k] for k in ("unsupported", "execution_failure")
        )
        and all(ratios[k] <= rule["maximum_" + k + "_ratio"] for k in ratios)
    )
    lifecycle = load(output / "lifecycle.json") if (output / "lifecycle.json").exists() else None
    lifecycle_ok = bool(lifecycle and (
        lifecycle.get("session_sha256") == file_hash(output / "session.json")
        and lifecycle.get("cost_kind") == "provider_billed"
        and (output / "billing-receipt.json").is_file()
        and lifecycle.get("billing_receipt_sha256") == file_hash(output / "billing-receipt.json")
        and all(type(lifecycle.get(k)) in {int, float}
                and math.isfinite(lifecycle[k]) and lifecycle[k] >= 0
                for k in ("actual_spend_usd", "gpu_lifetime_seconds"))
        and lifecycle["gpu_lifetime_seconds"] > 0
        and set(lifecycle.get("startup_seconds_by_arm", {})) == set(ARMS)
        and all(type(v) in {int, float} and math.isfinite(v) and v >= 0
                for v in lifecycle["startup_seconds_by_arm"].values())
    ))
    result = {
        "decision": ("inconclusive_missing_lifecycle_cost" if not lifecycle_ok else
                     "provisional_pilot_signal" if positive else "threshold_not_met"),
        "answer_and_episode_resource_thresholds_met": positive,
        "wins": wins,
        "losses": losses,
        "ties": len(cases) - wins - losses,
        "newly_unsupported_questions": newly_unsupported,
        "counts": counts,
        "resource_ratios": ratios,
        "pairs": pairs,
        "reviewer_kind": review["reviewer_kind"],
        "independent_human_review": review.get("independent_human_review"),
        "actual_spend_usd": lifecycle["actual_spend_usd"] if lifecycle_ok else None,
        "gpu_lifetime_seconds": lifecycle["gpu_lifetime_seconds"] if lifecycle_ok else None,
        "startup_seconds_by_arm": lifecycle["startup_seconds_by_arm"] if lifecycle_ok else None,
        "limitation": "Twelve papers, one seed; no significance or general improvement claim. "
        "Actual spend and lifecycle receipts must be reported separately.",
    }
    write(output / "comparison.json", result)
    return result


async def load_backend(output, arm, engine):
    """On GPU host: hash local adapter, use native loader, record hardware and path."""
    import torch
    from prime_rl.configs.shared import ClientConfig
    from prime_rl.orchestrator.clients import AdminPlane, load_lora_adapter

    output = Path(output).resolve()
    session, _ = validate_session(output)
    check_native_sources()
    if arm not in ARMS or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one CUDA GPU and a known frozen evaluation arm required")
    versions = {
        n: metadata.version(n)
        for n in ("prime-rl", "verifiers", "renderers", "transformers", "torch", "vllm")
    }
    if versions != {
        "prime-rl": "0.10.0",
        "verifiers": "0.4.0",
        "renderers": "0.1.13.dev5",
        "transformers": "5.6.2",
        "torch": "2.13.0+cu130",
        "vllm": "0.31.0",
    }:
        raise ValueError("use the pinned smoke GPU stack")
    plane = AdminPlane(
        ClientConfig(base_url=loopback_engine(engine) + "/v1", api_key_var="",
                     wait_for_ready_timeout=120)
    )
    # Keep native loading, while giving its local admin connection the same
    # no-proxy/no-credentials boundary as generation. No request has been sent.
    for client in plane.clients:
        await client.aclose()
    plane.clients = [httpx.AsyncClient(base_url=loopback_engine(engine),
                                     trust_env=False, follow_redirects=False, timeout=10)]
    start = time.monotonic()
    try:
        await plane.wait_for_ready("envoy-base")
        await load_lora_adapter(plane, "envoy", output / arm)
        model_root(engine, str(output / arm))
        validate_session(output)
        write(
            output / arm / "backend.json",
            {
                "session_sha256": file_hash(output / "session.json"),
                "arm": arm,
                "adapter_sha256": session["arms"][arm],
                "adapter_root": str(output / arm),
                "load_rpc_completed": True,
                "engine_tensor_attestation": False,
                "gpu": torch.cuda.get_device_name(0),
                "versions": versions,
                "adapter_load_seconds": time.monotonic() - start,
            },
        )
    finally:
        for client in plane.clients:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "freeze",
            "open-references",
            "collect-arm",
            "export-review",
            "summarize",
            "load-arm",
        ],
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--engine")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--backend-receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        if not args.training:
            parser.error("--training required")
        freeze(args.training, args.output)
    elif args.mode == "open-references":
        open_references(args.output)
    elif args.mode == "export-review":
        export_review(args.output)
    elif args.mode == "summarize":
        summarize(args.output)
    elif args.mode == "load-arm":
        if not args.arm or not args.engine:
            parser.error("--arm and --engine required")
        asyncio.run(load_backend(args.output, args.arm, args.engine))
    else:
        if not all((args.arm, args.engine, args.tokenizer, args.backend_receipt)):
            parser.error("--arm, --engine, --tokenizer and --backend-receipt required")
        def stop_requested(signum, frame):
            raise KeyboardInterrupt("termination requested; cleaning up evaluation containers")

        previous = signal.signal(signal.SIGTERM, stop_requested)
        try:
            asyncio.run(
                run_arm(args.output, args.arm, args.engine, args.tokenizer, args.backend_receipt)
            )
        finally:
            signal.signal(signal.SIGTERM, previous)
            from src.eval.reader_correction_replay import cleanup

            tracked = []
            for path in (args.output / args.arm / "rollouts").glob("*/owned-containers.json"):
                tracked.extend(load(path))
            proof = cleanup(tracked) if tracked else []
            if (args.output / args.arm).is_dir():
                write(args.output / args.arm / "controller-cleanup.json", {
                    "containers": proof, "all_absent": all(p["absent"] for p in proof)})
            if any(not p["absent"] for p in proof):
                raise RuntimeError("owned evaluation container cleanup could not be verified")


if __name__ == "__main__":
    main()

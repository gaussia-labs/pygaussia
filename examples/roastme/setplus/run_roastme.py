from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapter import KapsoWebhookAssistant, ReplayAssistant
from configuration import LoadedConfiguration, load_configuration
from enumerator import ConfigEntityEnumerator
from env import load_env
from grader import ConfigRuleGrader
from reporter import build_findings_reporter
from transforms import build_transforms

from gaussia.generators.roastme.dataset import to_dataset
from gaussia.generators.roastme.probes.catalogue import validate_catalogue
from gaussia.generators.roastme.probes.enumeration import EnumerationProbeEngine
from gaussia.generators.roastme.probes.library import ProbeLibrary
from gaussia.generators.roastme.profiler import Profiler
from gaussia.generators.roastme.reporting import render_findings_markdown
from gaussia.schemas.roastme import (
    BehavioralContract,
    Document,
    Principle,
    Probe,
    ProfilerResult,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_ENV_PATH = ROOT / ".env"


def build_contract(config: LoadedConfiguration) -> BehavioralContract:
    grader = ConfigRuleGrader(config.grader)
    return BehavioralContract(
        principles=[
            Principle(
                id=principle.id,
                weight=principle.weight,
                rubric=principle.rubric,
                grader=grader,
            )
            for principle in config.contract.principles
        ]
    )


def build_probes(
    config: LoadedConfiguration,
    contract: BehavioralContract,
    documents: list[Document],
) -> list[Probe]:
    transforms = build_transforms(config.transforms.literal_transforms)
    engine = EnumerationProbeEngine(
        ConfigEntityEnumerator(config.entities),
        entity_kinds=set(config.entities),
        transforms=transforms,
    )
    validate_catalogue(
        config.catalogue,
        contract,
        [engine],
        transforms=transforms,
    )
    return ProbeLibrary([engine]).generate(documents, config.catalogue)


def load_documents(config: LoadedConfiguration) -> list[Document]:
    return [
        Document(
            id=path.name,
            content=path.read_text(encoding="utf-8"),
            structured=True,
        )
        for path in sorted(config.corpus_dir.glob("*.md"))
    ]


def railway_api_token(config: LoadedConfiguration) -> str:
    target = config.evaluation.target
    result = subprocess.run(
        [
            "railway",
            "variable",
            "list",
            "--project",
            target.railway_project,
            "--service",
            target.railway_service,
            "--environment",
            target.railway_environment,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["API_TOKEN"]


def railway_channel_secrets(config: LoadedConfiguration) -> tuple[str, str]:
    evaluation = config.evaluation
    target = evaluation.target
    requested = json.dumps(
        {
            "assistant_id": evaluation.assistant_id,
            "webhook_secret": target.webhook_secret,
            "channel_account_secret": target.channel_account_secret,
        },
        separators=(",", ":"),
    )
    remote_script = (
        "import base64,json,os;"
        "from alquimia.registry.registry import AlquimiaRegistry;"
        "requested=json.loads(os.environ['ROASTME_REGISTRY_SECRETS']);"
        "registry=AlquimiaRegistry();"
        "registry.load({'agentspace_id':'default'});"
        "assistant_id=requested['assistant_id'];"
        "values={key:registry.current_agentspace.get_secret("
        "registry.get_secret(requested[key]),assistant_id) "
        "for key in ('webhook_secret','channel_account_secret')};"
        "print(base64.b64encode(json.dumps(values).encode()).decode())"
    )
    result = subprocess.run(
        [
            "railway",
            "ssh",
            "--project",
            target.railway_project,
            "--environment",
            target.railway_environment,
            "--service",
            target.railway_service,
            "env",
            f"ROASTME_REGISTRY_SECRETS={requested}",
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            remote_script,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(base64.b64decode(result.stdout.strip().splitlines()[-1]))
    return values["webhook_secret"], values["channel_account_secret"]


def build_live_target(
    config: LoadedConfiguration,
    env_path: Path,
) -> KapsoWebhookAssistant:
    actor_context = _read_json(_environment_path("ROASTME_ACTOR_CONTEXT_PATH", env_path))
    actor_offset = int(os.environ.get("ROASTME_ACTOR_OFFSET", "0"))
    api_token = os.environ.get("ROASTME_TARGET_API_TOKEN") or railway_api_token(config)
    webhook_secret, channel_account_id = railway_channel_secrets(config)
    return KapsoWebhookAssistant(
        target=config.evaluation.target,
        audit=config.evaluation.tool_audit,
        assistant_id=config.evaluation.assistant_id,
        api_token=api_token,
        webhook_secret=webhook_secret,
        channel_account_id=channel_account_id,
        actor_subjects=actor_context["actor_subjects"][actor_offset:],
    )


def persist_run(
    config: LoadedConfiguration,
    target: KapsoWebhookAssistant | ReplayAssistant,
    probes: list[Probe],
    result: ProfilerResult,
    session_id: str,
    target_mode: str,
    replayed_from: Path | None,
) -> Path:
    run_dir = config.output_dir / session_id
    run_dir.mkdir(parents=True)
    evaluation = config.evaluation
    dataset = to_dataset(
        probes,
        result.outcomes,
        session_id=session_id,
        assistant_id=evaluation.assistant_id,
        context=evaluation.context,
        language=evaluation.language,
    )
    (run_dir / "dataset.json").write_text(
        dataset.model_dump_json(indent=2, serialize_as_any=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "profile.json").write_text(
        result.profile.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "probes.json", [probe.model_dump() for probe in probes])
    audits = target.audits
    _write_json(run_dir / "transport-audit.json", audits)
    _write_json(
        run_dir / "summary.json",
        _summary(
            config,
            result,
            audits,
            session_id,
            target_mode,
            replayed_from,
        ),
    )
    return run_dir


def persist_findings(
    config: LoadedConfiguration,
    contract: BehavioralContract,
    documents: list[Document],
    probes: list[Probe],
    result: ProfilerResult,
    target: KapsoWebhookAssistant | ReplayAssistant,
    run_dir: Path,
    session_id: str,
) -> Path | None:
    reporter = build_findings_reporter(config.evaluation.reporting)
    if reporter is None:
        return None
    report = reporter.interpret(
        session_id=session_id,
        assistant_id=config.evaluation.assistant_id,
        target_model=config.evaluation.target.response_model,
        contract=contract,
        probes=probes,
        result=result,
        documents=documents,
        grader_context=config.grader.model_dump(mode="json"),
        transport_audit=target.audits,
    )
    (run_dir / "findings.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    findings_path = run_dir / "FINDINGS.md"
    findings_path.write_text(render_findings_markdown(report), encoding="utf-8")
    return findings_path


def _summary(
    config: LoadedConfiguration,
    result: ProfilerResult,
    audits: list[dict[str, Any]],
    session_id: str,
    target_mode: str,
    replayed_from: Path | None,
) -> dict[str, Any]:
    exposed = {tool for audit in audits for tool in audit["exposed_tools"]}
    invoked = {tool for audit in audits for tool in audit["invoked_tools"]}
    tool_audit = config.evaluation.tool_audit
    mutation_tools_exposed = sorted(exposed & tool_audit.mutating_tools)
    mutation_tools_invoked = sorted(invoked & tool_audit.mutating_tools)
    unexpected_tools_exposed = sorted(
        exposed - tool_audit.allowed_tools - tool_audit.mutating_tools
    )
    return {
        "session_id": session_id,
        "target_mode": target_mode,
        "replayed_from": str(replayed_from.resolve()) if replayed_from else None,
        "gaussia_source_commit": config.evaluation.gaussia_source_commit,
        "assistant_id": config.evaluation.assistant_id,
        "probe_count": len(result.outcomes),
        "scoreable": result.n_scoreable,
        "ungraded": result.n_ungraded,
        "overall_rate": result.overall_rate,
        "controls": sum(outcome.scoreable is False for outcome in result.outcomes),
        "control_violations": [
            outcome.probe_id
            for outcome in result.outcomes
            if outcome.scoreable is False and (outcome.violation or 0.0) > 0.0
        ],
        "tool_errors": sum(len(audit["tool_errors"]) for audit in audits),
        "exposed_tools": sorted(exposed),
        "invoked_tools": sorted(invoked),
        "unexpected_tools_exposed": unexpected_tools_exposed,
        "mutation_tools_exposed": mutation_tools_exposed,
        "mutation_tools_invoked": mutation_tools_invoked,
        "read_only_enforced": not (unexpected_tools_exposed or mutation_tools_exposed or mutation_tools_invoked),
    }


def _environment_path(name: str, env_path: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing {name}; copy .env.example to {env_path}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (env_path.parent / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    env_path = args.env.resolve()
    load_env(env_path)
    config = load_configuration(args.config_dir)
    contract = build_contract(config)
    documents = load_documents(config)
    probes = build_probes(config, contract, documents)
    if args.replay:
        target: KapsoWebhookAssistant | ReplayAssistant = ReplayAssistant.from_run(args.replay.resolve())
        target_mode = "replay"
    else:
        target = build_live_target(config, env_path)
        target_mode = "live"

    result = Profiler(contract, target).profile(probes)
    suffix = "-replay" if args.replay else ""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{config.evaluation.session_prefix}-{timestamp}{suffix}"
    run_dir = persist_run(
        config,
        target,
        probes,
        result,
        session_id,
        target_mode,
        args.replay,
    )
    findings_path = (
        None
        if args.skip_report
        else persist_findings(
            config,
            contract,
            documents,
            probes,
            result,
            target,
            run_dir,
            session_id,
        )
    )
    print((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"artifacts={run_dir}")
    if findings_path:
        print(f"findings={findings_path}")
    elif not args.skip_report and config.evaluation.reporting.enabled:
        print(f"findings=skipped (missing {config.evaluation.reporting.api_key_env})")


if __name__ == "__main__":
    main()

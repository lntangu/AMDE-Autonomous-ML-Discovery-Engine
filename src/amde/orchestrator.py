"""Run orchestration: setup, baselines, search, persistence, resume, stop.

This is the production coordinator. It composes the typed search spaces,
process-isolated evaluator, sealed challenge boundary, SQLite ledger,
checksummed checkpoints, resource controls, and the single-instance lock.
"""

from __future__ import annotations

import copy
import logging
import math
from logging.handlers import RotatingFileHandler
import random
import secrets
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .baselines.registry import BASELINE_NAMES, make_baseline_genome
from .benchmarks.arc import ArcAdapter, arc_status
from .benchmarks.registry import build_manifests, manifest_summary
from .config import Config, ResourcePlan, ResourcePolicy
from .discovery.factory import safety_kwargs
from .discovery.archive import DiscoveryArchive
from .discovery.genome import Genome
from .errors import (
    AMDEBaseError,
    CheckpointError,
    ConfigError,
    InfrastructureError,
    ResourcePressure,
    SealedProtocolError,
    StopRequested,
)
from .evaluation.evaluator import Evaluator, EvalRequest, aggregate_summary
from .evaluation.metrics import METRIC_HASH, METRIC_VERSION
from .evaluation.sealed import SealedEvaluator
from .runtime.heartbeat import HeartbeatWriter, StopController
from .runtime.lock import SingleInstanceLock, default_pid_alive
from .runtime.resources import HardwareInfo, ResourceSampler, detect_hardware, set_blas_threads
from .search.meta_policy import MetaPolicy
from .search.open_ended import behaviour_signature as genome_behaviour_signature
from .search.population import ArchiveManager, Individual, Population
from .search.portfolio import Portfolio
from .search.selection import choose_baseline, evaluate_promotion
from .search.stagnation import analyze_stagnation
from .search.strategies import STRATEGIES, Proposer
from .storage.artifacts import ArtifactStore
from .storage.budget import RunStorageBudget
from .storage.checkpoint import CheckpointManager
from .storage.ledger import Ledger
from .util import (
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    hash_json,
    human_duration,
    read_json,
    sanitize_id,
    seed_from,
    search_seed_for,
    utc_now,
    utc_stamp,
)

LIMITATIONS = [
    "Benchmarks are deterministic, locally generated synthetic/classical tasks, not a measure of general intelligence.",
    "The trainer is a small standard-library linear/tiny-MLP implementation; it does not represent modern deep learning.",
    "The optional PyTorch CUDA backend is used only after an actual tensor smoke test and resource checks; otherwise the reference CPU trainer is used.",
    "The sealed challenge is a single aggregate held-out opening, not a publication-grade generalization study.",
    "The promotion guard is a transparent heuristic with three fixed seeds, not a power analysis.",
    "ARC is blocked unless valid local task JSON exists; even then this engine is baseline-only for ARC grids.",
    "Long-running search can overfit the fixed generated suite; scores are not transferable claims.",
]


def new_run_id(mode: str) -> str:
    return sanitize_id(f"{mode}_{utc_stamp()}_{secrets.token_hex(3)}")


SMOKE_BASELINE_NAMES = tuple(name for name in BASELINE_NAMES if name not in {"mlp_medium", "linear_adagrad", "linear_nadam"})


def smoke_config(config: Config) -> Config:
    """Return a quick, one-generation smoke profile that exercises A-D.

    Smoke is a wiring/lifecycle check, not a scientific benchmark run. Keep a
    small representative development slice and one confirmation seed so the
    smoke command remains bounded as the research task zoo grows. Full
    campaigns continue to use the complete dev suite and configured seeds.
    """

    data = config.effective_copy()
    data["search"]["levels"] = ["A", "B", "C", "D"]
    data["search"]["open_level_after_generation"] = {"B": 0, "C": 0, "D": 0}
    data["search"]["proposals_per_generation"] = len(STRATEGIES)  # exercise every proposal arm once
    data["search"]["confirm_per_generation"] = 1
    quick_ids = {str(item) for item in data["benchmarks"].get("quick_task_ids", [])}
    # Keep every development quick task plus the sealed holdout declarations;
    # holdout data remains inaccessible to smoke/search regardless.
    data["benchmarks"]["task_specs"] = [
        spec for spec in data["benchmarks"]["task_specs"]
        if bool(spec.get("holdout_only", False)) or str(spec.get("task_id")) in quick_ids
    ]
    data["evaluation"]["confirm"]["seed_count"] = min(2, len(data["evaluation"].get("seeds", [1, 2])))
    # A CPU smoke should stay very short. CUDA smoke runs a fresh isolated
    # worker which imports PyTorch and initializes a CUDA context on Windows;
    # the old 10-second cap could expire during startup before training began.
    # Keep explicit CPU smoke fast, but grant auto/CUDA smoke bounded startup
    # headroom while preserving the overall 120-second smoke budget.
    requested_device = str(data.get("runtime", {}).get("device", "auto")).lower()
    cuda_capable_smoke = requested_device in {"auto", "cuda"}
    data["evaluation"]["quick"]["timeout_seconds"] = (
        max(45, int(data["evaluation"]["quick"].get("timeout_seconds", 10)))
        if cuda_capable_smoke else 10
    )
    data["evaluation"]["quick"]["max_workers"] = 1
    data["evaluation"]["confirm"]["timeout_seconds"] = 60 if cuda_capable_smoke else 10
    data["runtime"]["max_transient_retries"] = 0
    data["challenge"]["enabled"] = False
    data["budgets"]["smoke_max_generations"] = 1
    data["budgets"]["smoke_max_seconds"] = min(
        120, int(data["budgets"].get("smoke_max_seconds", 120))
    )
    return Config(data, config.project_root, config.source_path)


def _rng_state(rng: random.Random) -> list[Any]:
    state = rng.getstate()
    return [int(state[0]), [int(value) for value in state[1]], state[2]]


def _restore_rng(rng: random.Random, state: list[Any]) -> None:
    rng.setstate((int(state[0]), tuple(int(value) for value in state[1]), state[2]))


def _json_field(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if isinstance(value, str):
        try:
            import json

            return json.loads(value)
        except Exception:
            return None
    return value


def _individual_summary(individual: Individual | None) -> dict[str, Any] | None:
    if individual is None:
        return None
    return {
        "candidate_hash": individual.hash,
        "level": individual.level,
        "generation": individual.generation,
        "operator": individual.operator,
        "parents": list(individual.parents),
        "quick_score": individual.quick_score,
        "confirm_score": individual.confirm_score,
        "complexity": individual.complexity,
        "genome": individual.genome.to_dict(),
    }


def _missing_result(stage: str) -> dict[str, Any]:
    return {
        "status": "error",
        "aggregate": None,
        "error_type": "infrastructure",
        "error_code": "missing_result",
        "message": f"no {stage} result was produced",
        "elapsed_seconds": 0.0,
        "peak_rss_mb": None,
        "exit_code": None,
        "worker_pid": None,
        "memory_cap_enforced": False,
        "cache_key": None,
        "cache_hit": False,
    }


@dataclass
class _GenerationStats:
    generation: int
    started_at: str = field(default_factory=utc_now)
    proposals: int = 0
    valid: int = 0
    invalid: int = 0
    confirmations: int = 0
    promotions: int = 0
    strategies: dict[str, int] = field(default_factory=dict)
    quick_scores: list[float] = field(default_factory=list)
    confirm_scores: list[float] = field(default_factory=list)
    deadline_hit: bool = False
    challenge: dict[str, Any] | None = None

    def strategy(self, name: str) -> None:
        self.strategies[name] = self.strategies.get(name, 0) + 1


class _RunSession:
    def __init__(
        self,
        engine: "Engine",
        *,
        run_id: str,
        mode: str,
        total_generations: int,
        max_seconds: float,
        max_trials: int,
        resume: bool,
        stop_callback: Callable[[], bool] | None,
        candidate_hash: str | None = None,
        isolate_stop: bool = False,
    ) -> None:
        self.engine = engine
        self.config = engine.config
        self.mode = mode
        self.run_id = run_id
        self.total_generations = int(total_generations)
        self.max_seconds = float(max_seconds)
        self.max_trials = int(max_trials)
        self.resume = bool(resume)
        self.extra_stop_callback = stop_callback
        self.isolate_stop = bool(isolate_stop)
        self.requested_candidate = candidate_hash or None
        self.run_dir = ensure_dir(self.config.run_dir(run_id))
        self.logger = engine._configure_logging(self.run_dir)
        self.final_status: str | None = None
        self.failure: BaseException | None = None
        self.started_at = utc_now()
        self.session_started = time.monotonic()
        self.deadline = self.session_started + self.max_seconds if self.max_seconds > 0 else None
        self.lock = SingleInstanceLock(
            self.config.path("runs_dir") / ".amde.lock",
            run_id=run_id,
            stale_seconds=float(self.config.get("runtime.lock_stale_seconds", 300)),
        )
        self.stop = StopController(
            self.run_dir,
            stop_name=str(self.config.get("paths.stop_file_name", "STOP")),
            global_runs_dir=None if self.isolate_stop else self.config.path("runs_dir"),
        )
        self.heartbeat = HeartbeatWriter(
            self.run_dir,
            run_id=run_id,
            interval_seconds=float(self.config.get("runtime.heartbeat_seconds", 2)),
        )
        self.ledger: Ledger | None = None
        self.discovery_archive: DiscoveryArchive | None = None
        self.artifacts: ArtifactStore | None = None
        self.checkpoint: CheckpointManager | None = None
        self.storage_guard: RunStorageBudget | None = None
        self.evaluator: Evaluator | None = None
        self.sealed: SealedEvaluator | None = None
        self.sampler: ResourceSampler | None = None
        self.data_manifest: dict[str, Any] | None = None
        self.split_manifest: dict[str, Any] | None = None
        self.manifest_hash = ""
        self.manifest_summary: dict[str, Any] = {}
        self.cohort = ""
        self.population: Population | None = None
        self.archives: ArchiveManager | None = None
        self.portfolio: Portfolio | None = None
        self.meta = MetaPolicy(self.config)
        self.meta_features: dict[str, Any] = {}
        self.proposer: Proposer | None = None
        self.rng = random.Random(search_seed_for(int(self.config.get("seed"))))
        self.generation = -1
        self.prior_elapsed = 0.0
        self.trials_started = 0
        self.trials_since_checkpoint = 0
        self.plateau = 0
        self.incumbent: Individual | None = None
        self.promoted = False
        self.baselines_completed = False
        self.challenge_attempted = False
        self.confirm_attempted: set[str] = set()
        self.baselines: list[dict[str, Any]] = []
        self.baseline: dict[str, Any] | None = None
        self.baseline_name = ""
        self.baseline_genome: Genome | None = None
        self.arc_profile: dict[str, Any] = {}
        self.quick_specs: list[dict[str, Any]] = []
        self.dev_specs: list[dict[str, Any]] = []
        self.challenge_specs: list[dict[str, Any]] = []
        self.quick_seeds: list[int] = []
        self.confirm_seeds: list[int] = []
        self.quick_timeout = 10.0
        self.confirm_timeout = 40.0
        self.quick_workers = 1
        self.confirm_workers = 1
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _heartbeat_payload(self, extra: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "generation": self.generation,
            "trials": self.trials_started,
            "device_effective": self.engine.plan.device_effective,
            "gpu_effective": self.engine.plan.gpu_effective,
            "cpu_workers": self.engine.plan.cpu_workers,
        }
        if self.sampler is not None:
            try:
                payload["resources"] = self.sampler.snapshot(force=False)
            except Exception:
                pass
        payload.update(extra)
        return payload

    def execute(self) -> dict[str, Any]:
        try:
            self.lock.acquire()
            self.heartbeat.write(self._heartbeat_payload({"status": "starting"}), force=True)
            self._open()
            self._prepare_protocol()
            self._load_or_initialize()
            self._persist_protocol_files()
            self._prepare_search()
            self._write_run_manifest()
            if self.stop_requested():
                # A stop requested during setup is honored before any baseline
                # evaluation. The final checkpoint below still makes the run
                # resumable once the STOP file is removed.
                self.final_status = "stopped"
                if self.stop.global_stop_file and self.stop.global_stop_file.exists():
                    self.stop.clear(clear_global=True, clear_run=False)
                self.logger.info("stop requested before baseline evaluation")
            else:
                self._ensure_baselines()
                self._save_checkpoint("baselines_ready")
                self._check_run_storage_budget(force=True)
                if self.mode == "challenge":
                    self._run_challenge_mode()
                else:
                    self._search_loop()
            if self.final_status is None:
                self.final_status = "completed"
        except StopRequested:
            if self.stop.global_stop_file and self.stop.global_stop_file.exists():
                self.stop.clear(clear_global=True, clear_run=False)
            self.final_status = "stopped"
        except ResourcePressure as exc:
            self.final_status = "paused"
            self.failure = exc
            self.logger.warning("paused by resource pressure: %s", exc)
        except KeyboardInterrupt:
            self.stop.clear(clear_global=True)
            self.final_status = "stopped"
            self.failure = None
            self.logger.warning("keyboard interrupt; saving final checkpoint")
        except BaseException as exc:  # preserve a final audit even on failure
            self.final_status = "failed"
            self.failure = exc
            self.logger.exception("run failed")
        finally:
            try:
                self._close()
            finally:
                # A finalization error must never strand the single-instance
                # lock and make a later resume appear to be a duplicate run.
                try:
                    self.lock.release()
                except Exception:
                    pass
        summary = self._summary or self._build_summary()
        if self.final_status == "failed" and self.failure is not None:
            raise self.failure
        return summary

    def _open(self) -> None:
        self.ledger = Ledger(self.run_dir / "ledger.sqlite3")
        self.discovery_archive = DiscoveryArchive(
            self.config.path("storage_dir") / "discovery_archive.sqlite3",
            max_candidates=int(self.config.get("search.global_archive_max_candidates", 100_000)),
            max_sightings=int(self.config.get("search.global_archive_max_sightings", 1_000_000)),
        )
        self.artifacts = ArtifactStore(
            self.config.artifact_dir() / self.run_id,
            ledger=self.ledger,
            run_id=self.run_id,
            max_total_bytes=int(
                min(
                    float(self.config.get("budgets.max_disk_gb", 2.0)),
                    float(self.engine.policy.data["disk"]["max_run_gb"]),
                )
                * (1024**3)
            ),
            max_artifact_bytes=int(self.config.get("budgets.max_artifact_bytes", 1048576)),
        )
        self.checkpoint = CheckpointManager(
            self.run_dir, self.ledger, self.run_id,
            max_backups=int(self.config.get("storage.max_checkpoint_backups", 5)),
        )
        max_run_gb = min(
            float(self.config.get("budgets.max_disk_gb", 6.0)),
            float(self.engine.policy.data["disk"]["max_run_gb"]),
        )
        self.storage_guard = RunStorageBudget(
            [self.run_dir, self.config.artifact_dir() / self.run_id],
            max_total_bytes=int(max_run_gb * (1024**3)),
            pause_fraction=float(self.config.get("storage.pause_at_fraction", 0.80)),
            check_interval_seconds=float(self.config.get("storage.size_check_interval_seconds", 10.0)),
        )
        self.evaluator = Evaluator(
            self.config,
            stop_callback=self.stop_requested,
            memory_cap_mb=self.engine.plan.hard_cap_mb,
            device=self.engine.plan.device_effective,
        )
        self.sealed = SealedEvaluator(
            self.config,
            memory_cap_mb=self.engine.plan.hard_cap_mb,
            device=self.engine.plan.device_effective,
        )
        self.sampler = ResourceSampler(
            self.engine.policy,
            self.config.project_root,
            gpu_active=bool(self.engine.plan.gpu_effective),
        )
        self.quick_specs = self._specs_by_ids(self.config.quick_task_ids())
        self.dev_specs = self.config.dev_task_specs()
        self.challenge_specs = self.config.challenge_task_specs()
        self.quick_seeds = self.config.eval_seeds(int(self.config.get("evaluation.quick.seed_count", 1)))
        self.confirm_seeds = self.config.eval_seeds(int(self.config.get("evaluation.confirm.seed_count", 3)))
        self.quick_timeout = float(self.config.get("evaluation.quick.timeout_seconds", 10))
        self.confirm_timeout = float(self.config.get("evaluation.confirm.timeout_seconds", 40))
        worker_limit = 1 if self.engine.plan.gpu_effective else self.engine.plan.cpu_workers
        self.quick_workers = max(
            1,
            min(
                worker_limit,
                int(self.config.get("evaluation.quick.max_workers", 2)),
            ),
        )
        self.confirm_workers = max(
            1,
            min(
                worker_limit,
                int(self.config.get("evaluation.confirm.max_workers", 2)),
            ),
        )

    def _prepare_protocol(self) -> None:
        manifests = build_manifests(self.config)
        self.data_manifest = manifests["data_manifest"]
        self.split_manifest = manifests["split_manifest"]
        self.manifest_summary = manifest_summary(manifests)
        self.manifest_hash = str(self.data_manifest["manifest_hash"])
        self.cohort = str(manifests["cohort"])
        # The challenge examples returned by build_manifests are deliberately
        # discarded here. Only hashes in the manifests are persisted.
        manifests.pop("tasks", None)
        manifests.clear()
        profile = arc_status(self.config.arc_path())
        self.arc_profile = {"profile": profile}
        if profile.get("status") == "READY_BASELINE_ONLY":
            try:
                self.arc_profile["baseline"] = ArcAdapter(self.config.arc_path()).run_baseline(max_tasks=40)
            except Exception as exc:
                self.arc_profile["baseline_error"] = f"{type(exc).__name__}: {exc}"
        self.meta_features = self.meta.features(self.data_manifest)

    def _persist_protocol_files(self) -> None:
        ensure_dir(self.run_dir / "manifests")
        atomic_write_json(self.run_dir / "manifests" / "data_manifest.json", self.data_manifest)
        atomic_write_json(self.run_dir / "manifests" / "split_manifest.json", self.split_manifest)
        atomic_write_json(self.run_dir / "arc_profile.json", self.arc_profile)

    def _load_or_initialize(self) -> None:
        assert self.ledger is not None
        assert self.checkpoint is not None
        state = self.checkpoint.load() if self.resume else None
        if state is None:
            if self.ledger.get_run(self.run_id) is not None:
                raise ConfigError(
                    f"run {self.run_id!r} already exists but has no loadable checkpoint; "
                    "choose a new run id or restore a checkpoint backup"
                )
            self.ledger.upsert_run(
                run_id=self.run_id,
                mode=self.mode,
                engine_version=str(self.config.get("engine_version")),
                config_hash=self.config.config_hash,
                protocol_hash=self.config.protocol_hash,
                config_json=self.config.effective_copy(),
                manifest_hash=self.manifest_hash,
                status="running",
            )
            self.ledger.record_event(
                self.run_id,
                "run_initialized",
                payload={
                    "mode": self.mode,
                    "config_hash": self.config.config_hash,
                    "protocol_hash": self.config.protocol_hash,
                    "manifest_hash": self.manifest_hash,
                },
            )
            return

        if state.get("run_id") != self.run_id:
            raise CheckpointError("checkpoint belongs to a different run id")
        if state.get("config_hash") != self.config.config_hash:
            raise CheckpointError(
                "checkpoint config hash differs from the active config; refusing to resume"
            )
        if state.get("protocol_hash") != self.config.protocol_hash:
            raise CheckpointError(
                "checkpoint protocol hash differs from the active protocol; refusing to resume"
            )
        saved_manifest = state.get("manifest_summary", {})
        if saved_manifest.get("data_manifest_hash") != self.manifest_summary.get("data_manifest_hash"):
            raise CheckpointError("checkpoint data manifest differs from the regenerated manifest")
        if self.ledger.get_run(self.run_id) is None:
            raise CheckpointError("checkpoint exists but the run ledger is missing")

        safety = safety_kwargs(self.config)
        self.generation = int(state.get("generation", -1))
        self.trials_started = int(state.get("trials_started", 0))
        self.plateau = int(state.get("plateau", 0))
        self.promoted = bool(state.get("promoted", False))
        self.baselines_completed = bool(state.get("baselines_completed", False))
        self.challenge_attempted = bool(state.get("challenge_attempted", False))
        self.confirm_attempted = set(str(item) for item in state.get("confirm_attempted", []))
        self.prior_elapsed = float(state.get("elapsed_seconds", 0.0))
        self.started_at = str(state.get("started_at", self.started_at))
        self.population = Population.from_dict(state.get("population", {}), safety=safety)
        self.archives = ArchiveManager.from_dict(state.get("archives", {}))
        self.portfolio = Portfolio.from_dict(state.get("portfolio", {}))
        if state.get("rng_state"):
            _restore_rng(self.rng, state["rng_state"])
        incumbent = state.get("incumbent")
        self.incumbent = (
            Individual.from_dict(incumbent, safety=safety) if isinstance(incumbent, dict) else None
        )
        self.ledger.upsert_run(
            run_id=self.run_id,
            mode=self.mode,
            engine_version=str(self.config.get("engine_version")),
            config_hash=self.config.config_hash,
            protocol_hash=self.config.protocol_hash,
            config_json=self.config.effective_copy(),
            manifest_hash=self.manifest_hash,
            status="running",
        )
        interrupted = self.ledger.mark_interrupted_trials(self.run_id)
        self.ledger.record_event(
            self.run_id,
            "run_resumed",
            payload={
                "generation": self.generation,
                "interrupted_trials": interrupted,
                "recovered_from": state.get("_checkpoint_meta", {}),
                "mode": self.mode,
            },
        )
        self.logger.info(
            "resumed run %s at generation %s (%s interrupted trials marked)",
            self.run_id,
            self.generation,
            interrupted,
        )

    def _prepare_search(self) -> None:
        safety = safety_kwargs(self.config)
        if self.portfolio is None:
            strategies = [str(item) for item in self.config.get("search.strategies", list(STRATEGIES))]
            for strategy in strategies:
                if strategy not in STRATEGIES:
                    raise ConfigError(f"unknown search strategy: {strategy}")
            self.portfolio = Portfolio(
                strategies,
                exploration_floor=float(self.config.get("search.exploration_floor", 0.15)),
            )
        if self.population is None:
            self.population = Population(
                capacity=int(self.config.get("search.population_size", 8)),
                islands=int(self.config.get("search.islands", 2)),
                elite_size=int(self.config.get("search.elite_size", 2)),
                safety=safety,
            )
        if self.archives is None:
            self.archives = ArchiveManager()
        # Optional campaign genomes are warm-start seeds only. They carry no
        # cross-run scores and must earn fitness again on this run's protocol.
        seed_genomes = self.config.get("search.seed_genomes", [])
        if not isinstance(seed_genomes, list):
            raise ConfigError("search.seed_genomes must be a list of validated genome objects")
        seed_records: list[tuple[dict[str, Any], str]] = [
            (raw, "campaign_seed") for raw in seed_genomes if isinstance(raw, dict)
        ]
        if len(seed_records) != len(seed_genomes):
            raise ConfigError("search.seed_genomes entries must be objects")
        # Historical candidates are warm-start proposals only. Their prior
        # scores are never imported; every candidate is evaluated against the
        # new campaign's immutable protocol and fresh seeds.
        archive_limit = int(self.config.get("search.global_archive_seed_count", 4))
        discovery_archive = getattr(self, "discovery_archive", None)
        if archive_limit > 0 and discovery_archive is not None:
            known_hashes = {Genome(raw, **safety).hash for raw, _ in seed_records}
            for raw in discovery_archive.warm_start_genomes(limit=archive_limit):
                genome = Genome(raw, **safety)
                if genome.hash not in known_hashes:
                    seed_records.append((raw, "global_archive_seed"))
                    known_hashes.add(genome.hash)
        for seed_index, (raw_genome, source_operator) in enumerate(seed_records[: min(12, self.population.capacity)]):
            genome = Genome(raw_genome, **safety)
            self.population.add(
                Individual(
                    genome=genome,
                    generation=-1,
                    operator=source_operator,
                    parents=[],
                    island=seed_index % self.population.n_islands,
                    status="campaign_seed_pending",
                )
            )
        self.proposer = Proposer(
            population=self.population,
            archives=self.archives,
            rng=self.rng,
            levels=list(self.config.get("search.levels", ["A", "B", "C", "D"])),
            safety=safety,
        )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    def _ensure_baselines(self) -> None:
        assert self.ledger is not None
        assert self.evaluator is not None
        assert self.artifacts is not None
        complete = True
        records = []
        baseline_names = SMOKE_BASELINE_NAMES if self.mode == "smoke" else BASELINE_NAMES
        for name in baseline_names:
            self.lock.heartbeat()
            existing = self.ledger.get_baseline(name, self.config.protocol_hash, self.manifest_hash)
            if existing is None:
                complete = False
                existing = self._evaluate_baseline(name)
            records.append(existing)
        self.baselines = records
        self.baseline = choose_baseline(records)
        if self.baseline is None:
            raise InfrastructureError("no baseline produced a valid aggregate")
        self.baseline_name = str(self.baseline["name"])
        self.baseline_genome = make_baseline_genome(self.baseline_name)
        self.baselines_completed = True
        self.ledger.record_event(
            self.run_id,
            "baselines_ready",
            payload={
                "strongest": self.baseline_name,
                "means": {
                    str(item["name"]): float(item["aggregate"]["mean"])
                    for item in records
                    if item.get("aggregate")
                },
                "complete": complete,
            },
        )
        self.logger.info(
            "baselines ready; strongest=%s mean=%.4f",
            self.baseline_name,
            float(self.baseline["aggregate"]["mean"]),
        )

    def _evaluate_baseline(self, name: str) -> dict[str, Any]:
        assert self.ledger is not None
        assert self.evaluator is not None
        assert self.artifacts is not None
        genome = make_baseline_genome(name)
        request = self.evaluator.build_request(
            genome=genome,
            baseline=None if genome is not None else "majority_mean",
            stage="baseline",
            task_specs=self.dev_specs,
            seeds=self.confirm_seeds,
            epochs_scale=1.0,
            timeout_seconds=self.confirm_timeout,
            manifest_hash=self.manifest_hash,
        )
        result = self._evaluate_with_retries(request)
        artifact_path = None
        artifact_sha = None
        if result["status"] == "ok":
            artifact = self.artifacts.put_json(result["aggregate"], kind="baseline_aggregate")
            if artifact.stored:
                artifact_path = artifact.relative_path
                artifact_sha = artifact.sha256
        self.ledger.save_baseline(
            run_id=self.run_id,
            name=name,
            protocol_hash=self.config.protocol_hash,
            manifest_hash=self.manifest_hash,
            status="ok" if result["status"] == "ok" else result["status"],
            aggregate=result.get("aggregate"),
            seeds=self.confirm_seeds,
            metric_version=METRIC_VERSION,
            metric_hash=METRIC_HASH,
            elapsed_seconds=result.get("elapsed_seconds"),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha,
        )
        if self.stop_requested() and result["status"] != "ok":
            raise StopRequested(f"stop requested during baseline evaluation of {name}")
        if result["status"] != "ok":
            raise InfrastructureError(
                f"baseline {name} failed ({result['status']}): {result.get('message')}"
            )
        self.ledger.record_event(
            self.run_id,
            "baseline_evaluated",
            payload={
                "name": name,
                "mean": float(result["aggregate"]["mean"]),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
        )
        record = self.ledger.get_baseline(name, self.config.protocol_hash, self.manifest_hash)
        if record is None:
            raise InfrastructureError(f"baseline {name} was saved but cannot be read back")
        return record

    def _evaluate_with_retries(self, request: EvalRequest) -> dict[str, Any]:
        assert self.evaluator is not None
        attempts = 1 + max(0, int(self.config.get("runtime.max_transient_retries", 2)))
        result = None
        for attempt in range(attempts):
            result = self.evaluator.evaluate(request)
            if result["status"] == "ok":
                return result
            if self.stop_requested() and (result.get("error_code") == "stop_requested" or result["status"] == "interrupted"):
                return result
            if result["status"] in ("invalid", "protocol_error"):
                return result
            if attempt + 1 < attempts:
                delay = float(self.config.get("runtime.retry_backoff_seconds", 1.0)) * (attempt + 1)
                self.logger.warning(
                    "transient evaluation failure (%s); retrying in %.1fs: %s",
                    result.get("error_code"),
                    delay,
                    result.get("message"),
                )
                time.sleep(delay)
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Search loop
    # ------------------------------------------------------------------
    def _search_loop(self) -> None:
        while True:
            if self.total_generations > 0 and self.generation + 1 >= self.total_generations:
                self.final_status = "completed"
                return
            if self.stop_requested():
                self.final_status = "stopped"
                return
            if self._deadline_reached():
                self.final_status = "time_budget"
                return
            if self.max_trials > 0 and self.trials_started >= self.max_trials:
                self.final_status = "trial_budget"
                return
            self._check_resources()
            generation = self.generation + 1
            self.logger.info("starting generation %s", generation)
            self.heartbeat.write(
                self._heartbeat_payload({"status": "running", "generation": generation, "trials": self.trials_started}),
                force=True,
            )
            self._run_generation(generation)
            self.generation = generation
            self._save_checkpoint("generation_completed")
            # Check only after the generation checkpoint is durable so a storage
            # pressure pause resumes from a coherent state.
            self._check_run_storage_budget(force=True)
            self.heartbeat.write(
                self._heartbeat_payload(
                    {
                        "status": "running",
                        "generation": self.generation,
                        "trials": self.trials_started,
                        "best": _individual_summary(self.population.best()) if self.population else None,
                    }
                ),
                force=True,
            )
            if self._deadline_reached():
                self.final_status = "time_budget"
                return

    def _run_generation(self, generation: int) -> None:
        assert self.population is not None
        assert self.archives is not None
        assert self.portfolio is not None
        assert self.proposer is not None
        assert self.ledger is not None
        assert self.evaluator is not None
        stats = _GenerationStats(generation=generation)
        available = self._available_levels(generation)
        remaining = self.max_trials - self.trials_started if self.max_trials > 0 else 10**9
        configured = int(self.config.get("search.proposals_per_generation", 6))
        if self.mode == "smoke":
            configured = max(configured, len(STRATEGIES), len(available))
        proposal_count = max(0, min(configured, remaining))
        if proposal_count == 0:
            stats.deadline_hit = True
            self._record_generation(stats, "budget_exhausted")
            return

        proposals: list[dict[str, Any]] = []
        # Cross-round campaign genomes are evaluated directly before ordinary
        # proposals. They enter with no fitness and must earn it on this round's
        # freshly seeded protocol. If more than one generation's budget remains,
        # untested seeds stay pending for the next generation.
        campaign_seeds = [
            item for item in self.population.all()
            if item.operator in {"campaign_seed", "global_archive_seed"}
            and item.status == "campaign_seed_pending"
        ]
        campaign_seed_iter = iter(campaign_seeds[:proposal_count] if generation >= 0 else [])
        for index in range(proposal_count):
            try:
                individual = next(campaign_seed_iter)
                individual.generation = generation
                individual.status = "proposed"
                strategy = "campaign_seed"
                meta_note = {"mode": "cross_round_re_evaluation", "prior": None}
            except StopIteration:
                level_override = available[index % len(available)]
                if self.mode == "smoke":
                    strategy = STRATEGIES[index % len(STRATEGIES)]
                    meta_note = {"mode": "smoke_cycle", "prior": None}
                else:
                    strategy, meta_note = self.meta.choose(self.portfolio, self.rng, self.meta_features)
                individual = self.proposer.propose(
                    generation=generation,
                    strategy=strategy,
                    level_override=level_override,
                )
            self.population.seen.add(individual.hash)
            self.ledger.save_candidate(
                run_id=self.run_id,
                candidate_hash=individual.hash,
                genome=individual.genome.to_dict(),
                level=individual.level,
                complexity=individual.complexity,
                generation=generation,
                parent_hash=individual.parents[0] if individual.parents else None,
                operator=individual.operator,
            )
            request = self.evaluator.build_request(
                genome=individual.genome,
                baseline=None,
                stage="quick",
                task_specs=self.quick_specs,
                seeds=self.quick_seeds,
                epochs_scale=float(self.config.get("evaluation.quick.epochs_scale", 0.35)),
                timeout_seconds=self.quick_timeout,
                manifest_hash=self.manifest_hash,
            )
            stats.strategy(strategy)
            stats.proposals += 1
            proposals.append(
                {
                    "individual": individual,
                    "request": request,
                    "strategy": strategy,
                    "meta": meta_note,
                }
            )

        quick_results = self._run_quick_batch(proposals, generation, stats)
        valid = []
        for item, result in zip(proposals, quick_results):
            individual: Individual = item["individual"]
            reward = 0.0
            if result["status"] == "ok":
                aggregate = result["aggregate"]
                individual.quick_score = float(aggregate["mean"])
                individual.quick_aggregate = aggregate_summary(aggregate)
                individual.status = "quick_ok"
                self.population.add(individual)
                self._observe_global_discovery(individual, stage="quick", score=individual.quick_score)
                archive_info = self.archives.observe(individual)
                individual.novelty = float(archive_info["novelty"])
                self.population.update(individual)
                self._record_archive(individual, archive_info)
                stats.valid += 1
                stats.quick_scores.append(individual.quick_score)
                valid.append(individual)
                reward = max(0.0, min(1.0, individual.quick_score))
            else:
                stats.invalid += 1
            self.portfolio.update(item["strategy"], reward)

        if self.stop_requested():
            self._record_generation(stats, "stopped")
            raise StopRequested("stop requested during quick evaluation")
        if self._deadline_reached():
            stats.deadline_hit = True
            self._record_generation(stats, "time_budget")
            return
        if self.max_trials > 0 and self.trials_started >= self.max_trials:
            self._record_generation(stats, "budget_exhausted")
            return

        promoted_this_generation = self._confirm_phase(generation, stats, valid)
        if promoted_this_generation:
            self.plateau = 0
        else:
            self.plateau += 1

        if self.stop_requested():
            self._record_generation(stats, "stopped")
            raise StopRequested("stop requested during confirmation")
        if self._deadline_reached():
            stats.deadline_hit = True
            self._record_generation(stats, "time_budget")
            return

        if self._should_open_challenge(generation):
            challenge = self._open_challenge(generation, automatic=True)
            stats.challenge = {
                "status": challenge.get("status"),
                "delta": challenge.get("delta"),
                "opening_index": challenge.get("opening_index"),
            }
            self.challenge_attempted = True

        self._record_generation(stats, "completed")

    def _run_quick_batch(
        self,
        proposals: list[dict[str, Any]],
        generation: int,
        stats: _GenerationStats,
    ) -> list[dict[str, Any]]:
        assert self.ledger is not None
        assert self.evaluator is not None
        results: list[dict[str, Any] | None] = [None] * len(proposals)
        pending_requests: list[EvalRequest] = []
        pending_meta: list[dict[str, Any]] = []
        pending_by_cache_key: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(proposals):
            if index % 2 == 0:
                self.lock.heartbeat()
            request: EvalRequest = item["request"]
            cached = self._cached_result(request)
            if cached is not None:
                results[index] = cached
                self.ledger.record_event(
                    self.run_id,
                    "cache_hit",
                    level="DEBUG",
                    payload={"stage": "quick", "candidate": item["individual"].hash},
                )
                continue

            # Requests sharing a cache key have identical protocol-bound evaluation
            # identity. A lookup cannot see another request still running, so merge
            # them before dispatch to avoid redundant work and duplicate-success races.
            shared = pending_by_cache_key.get(request.cache_key)
            if shared is not None:
                shared["indices"].append(index)
                self.ledger.record_event(
                    self.run_id,
                    "batch_cache_deduplicated",
                    level="DEBUG",
                    payload={"stage": "quick", "candidate": item["individual"].hash},
                )
                continue

            trial_id = uuid.uuid4().hex
            self.ledger.begin_trial(
                run_id=self.run_id,
                trial_id=trial_id,
                generation=generation,
                candidate_hash=item["individual"].hash,
                stage="quick",
                baseline=None,
                seeds=self.quick_seeds,
                task_ids=self.quick_specs and [str(spec["task_id"]) for spec in self.quick_specs],
                cache_key=request.cache_key,
                protocol_hash=self.config.protocol_hash,
                manifest_hash=self.manifest_hash,
                metric_hash=METRIC_HASH,
                epochs_scale=request.epochs_scale,
            )
            self.trials_started += 1
            self.trials_since_checkpoint += 1
            meta = {"trial_id": trial_id, "item": item, "indices": [index]}
            pending_requests.append(request)
            pending_meta.append(meta)
            pending_by_cache_key[request.cache_key] = meta
        if pending_requests:
            raw = self.evaluator.evaluate_many(pending_requests, max_workers=self.quick_workers)
            for meta, result in zip(pending_meta, raw):
                self._finish_trial(result, meta["trial_id"], "quick")
                for result_index in meta["indices"]:
                    results[result_index] = dict(result)
        return [result or _missing_result("quick") for result in results]

    def _confirm_phase(
        self,
        generation: int,
        stats: _GenerationStats,
        valid: list[Individual],
    ) -> bool:
        assert self.ledger is not None
        assert self.evaluator is not None
        assert self.population is not None
        assert self.archives is not None
        if not valid:
            return False
        candidates = self._choose_confirms(valid)
        if not candidates:
            return False
        remaining = self.max_trials - self.trials_started if self.max_trials > 0 else 10**9
        candidates = candidates[: max(0, remaining)]
        if not candidates:
            return False
        requests = []
        for individual in candidates:
            self.confirm_attempted.add(individual.hash)
            requests.append(
                self.evaluator.build_request(
                    genome=individual.genome,
                    baseline=None,
                    stage="confirm",
                    task_specs=self.dev_specs,
                    seeds=self.confirm_seeds,
                    epochs_scale=1.0,
                    timeout_seconds=self.confirm_timeout,
                    manifest_hash=self.manifest_hash,
                )
            )
        results: list[dict[str, Any] | None] = [None] * len(candidates)
        pending_requests: list[EvalRequest] = []
        pending_meta: list[dict[str, Any]] = []
        pending_by_cache_key: dict[str, dict[str, Any]] = {}
        for index, (individual, request) in enumerate(zip(candidates, requests)):
            self.lock.heartbeat()
            cached = self._cached_result(request)
            if cached is not None:
                results[index] = cached
                continue
            shared = pending_by_cache_key.get(request.cache_key)
            if shared is not None:
                shared["indices"].append(index)
                self.ledger.record_event(
                    self.run_id,
                    "batch_cache_deduplicated",
                    level="DEBUG",
                    payload={"stage": "confirm", "candidate": individual.hash},
                )
                continue
            trial_id = uuid.uuid4().hex
            self.ledger.begin_trial(
                run_id=self.run_id,
                trial_id=trial_id,
                generation=generation,
                candidate_hash=individual.hash,
                stage="confirm",
                baseline=None,
                seeds=self.confirm_seeds,
                task_ids=[str(spec["task_id"]) for spec in self.dev_specs],
                cache_key=request.cache_key,
                protocol_hash=self.config.protocol_hash,
                manifest_hash=self.manifest_hash,
                metric_hash=METRIC_HASH,
                epochs_scale=request.epochs_scale,
            )
            self.trials_started += 1
            self.trials_since_checkpoint += 1
            meta = {"trial_id": trial_id, "individual": individual, "indices": [index]}
            pending_requests.append(request)
            pending_meta.append(meta)
            pending_by_cache_key[request.cache_key] = meta
        if pending_requests:
            raw = self.evaluator.evaluate_many(pending_requests, max_workers=self.confirm_workers)
            for meta, result in zip(pending_meta, raw):
                self._finish_trial(result, meta["trial_id"], "confirm")
                for result_index in meta["indices"]:
                    results[result_index] = dict(result)

        promoted = False
        assert self.baseline is not None
        for individual, result in zip(candidates, results):
            if result is None or result["status"] != "ok":
                assert result is not None
                self.ledger.save_decision(
                    run_id=self.run_id,
                    generation=generation,
                    candidate_hash=individual.hash,
                    decision="confirm_failed",
                    reason=f"confirmation status {result['status']}: {result.get('message', '')}",
                    evidence={
                        "stage": "confirm",
                        "status": result["status"],
                        "error_type": result.get("error_type"),
                        "error_code": result.get("error_code"),
                    },
                )
                continue
            aggregate = result["aggregate"]
            stats.confirmations += 1
            stats.confirm_scores.append(float(aggregate["mean"]))
            self.population.update_confirm(individual.hash, aggregate)
            updated = self.population.members.get(individual.hash, individual)
            self._observe_global_discovery(updated, stage="confirm", score=float(aggregate["mean"]))
            archive_info = self.archives.observe(updated)
            self._record_archive(updated, archive_info)
            decision = evaluate_promotion(
                candidate=aggregate,
                baseline=self.baseline["aggregate"],
                config=dict(self.config.get("evaluation.promotion", {})),
                candidate_hash=individual.hash,
                baseline_name=self.baseline_name,
            )
            label = decision["decision"]
            evidence = dict(decision["evidence"])
            evidence["incumbent_hash"] = self.incumbent.hash if self.incumbent else None
            evidence["incumbent_score"] = self.incumbent.confirm_score if self.incumbent else None
            if label == "promote":
                if self._is_new_incumbent(updated):
                    self.incumbent = updated
                    self.promoted = True
                    promoted = True
                    label = "promote"
                    self.ledger.add_archive(
                        run_id=self.run_id,
                        kind="promotion",
                        candidate_hash=updated.hash,
                        reason=decision["reason"],
                        payload=aggregate_summary(aggregate) or {},
                    )
                else:
                    label = "eligible_not_incumbent"
            self.ledger.save_decision(
                run_id=self.run_id,
                generation=generation,
                candidate_hash=individual.hash,
                decision=label,
                reason=decision["reason"],
                evidence=evidence,
            )
            stats.promotions += 1 if label == "promote" else 0
        return promoted

    def _observe_global_discovery(self, individual: Individual, *, stage: str, score: float | None) -> None:
        """Persist successful candidates across campaigns without affecting search decisions."""
        if self.discovery_archive is None:
            return
        try:
            result = self.discovery_archive.observe(
                individual.genome,
                run_id=self.run_id,
                generation=individual.generation,
                operator=individual.operator,
                stage=stage,
                score=score,
                protocol_hash=self.config.protocol_hash,
                behavioural_signature=(
                    genome_behaviour_signature(individual.genome)
                    if individual.genome.data["optimizer"]["kind"] == "dsl" else None
                ),
                metadata={"island": individual.island, "level": individual.level},
            )
            if not result.get("archived", True):
                self.logger.warning("global discovery archive capacity reached; candidate was not archived")
                if self.ledger is not None:
                    self.ledger.record_event(
                        self.run_id, "global_discovery_archive_capacity", level="WARNING",
                        payload={"candidate_hash": individual.hash, "stage": stage},
                    )
        except Exception as exc:
            # Per-run SQLite ledger remains authoritative; a secondary global
            # archive failure must be visible but must not destroy a valid run.
            self.logger.warning("global discovery archive write failed: %s: %s", type(exc).__name__, exc)
            if self.ledger is not None:
                try:
                    self.ledger.record_event(
                        self.run_id, "global_discovery_archive_error", level="WARNING",
                        payload={"error_type": type(exc).__name__, "message": str(exc)},
                    )
                except Exception:
                    pass

    def _choose_confirms(self, valid: list[Individual]) -> list[Individual]:
        assert self.population is not None
        limit = int(self.config.get("search.confirm_per_generation", 2))
        candidates = [
            individual
            for individual in valid
            if individual.hash not in self.confirm_attempted
            and (
                self.population.members.get(individual.hash) is None
                or self.population.members[individual.hash].confirm_score is None
            )
        ]
        # The same genome may be emitted by more than one proposal operator.
        # A confirmation is attached to genome identity, not proposal occurrence.
        unique_candidates: dict[str, Individual] = {}
        for individual in candidates:
            unique_candidates.setdefault(individual.hash, individual)
        candidates = list(unique_candidates.values())
        if not candidates:
            return []
        by_level: dict[str, list[Individual]] = {}
        for individual in candidates:
            by_level.setdefault(individual.level, []).append(individual)
        level_winners = [
            max(items, key=lambda item: (item.quick_score or -1.0))
            for items in by_level.values()
        ]
        ordered = sorted(level_winners, key=lambda item: (item.quick_score or -1.0), reverse=True)
        remaining = sorted(candidates, key=lambda item: (item.quick_score or -1.0), reverse=True)
        for individual in remaining:
            if individual not in ordered:
                ordered.append(individual)
        return ordered[:limit]

    def _is_new_incumbent(self, candidate: Individual) -> bool:
        if self.incumbent is None:
            return True
        if candidate.confirm_score is None:
            return False
        incumbent_score = float(self.incumbent.confirm_score or -1.0)
        candidate_score = float(candidate.confirm_score)
        if candidate_score > incumbent_score + 1e-12:
            return True
        if abs(candidate_score - incumbent_score) <= 1e-12:
            return candidate.complexity < self.incumbent.complexity
        return False

    # ------------------------------------------------------------------
    # Challenge
    # ------------------------------------------------------------------
    def _should_open_challenge(self, generation: int) -> bool:
        if not bool(self.config.get("challenge.enabled", False)):
            return False
        # Challenge outputs must not be revealed automatically in a run that
        # continues optimizing. Repeated runs share the deterministic cohort, so
        # observing it mid-search can leak the holdout into human model selection.
        if not bool(self.config.get("challenge.auto_open_during_search", False)):
            return False
        if self.mode == "smoke" or self.challenge_attempted:
            return False
        assert self.ledger is not None
        if self.ledger.count_challenge_openings(self.run_id) >= int(
            self.config.get("challenge.max_openings_per_run", 1)
        ):
            return False
        minimum = int(self.config.get("challenge.min_generations_between_openings", 5))
        if generation + 1 < minimum:
            return False
        return self.incumbent is not None

    def _open_challenge(
        self,
        generation: int,
        *,
        automatic: bool,
        candidate_hash: str | None = None,
    ) -> dict[str, Any]:
        assert self.ledger is not None
        assert self.sealed is not None
        assert self.population is not None
        max_openings = int(self.config.get("challenge.max_openings_per_run", 1))
        if self.ledger.count_challenge_openings(self.run_id) >= max_openings:
            return {
                "status": "exhausted",
                "message": f"max challenge openings reached ({max_openings})",
            }
        candidate = self._resolve_candidate(candidate_hash)
        if candidate is None:
            return {"status": "no_candidate", "message": "no confirmed candidate is available"}
        seeds = [int(seed) for seed in self.config.get("challenge.seeds", [])]
        if not seeds:
            seeds = self.config.eval_seeds(1)
        baseline_genome = make_baseline_genome(self.baseline_name)
        opening_index = self.ledger.count_challenge_openings(self.run_id) + 1
        result = self.sealed.open_opening(
            candidate=candidate.genome,
            baseline_name=self.baseline_name,
            baseline_genome=baseline_genome,
            task_specs=self.challenge_specs,
            seeds=seeds,
            opening_index=opening_index,
            cohort=self.cohort,
            stop_callback=self.stop_requested,
        )
        if result["status"] == "ok":
            self.ledger.save_challenge_opening(
                {
                    "run_id": self.run_id,
                    "opening_index": opening_index,
                    "cohort": self.cohort,
                    "candidate_hash": candidate.hash,
                    "candidate_aggregate": result["candidate"],
                    "baseline_name": self.baseline_name,
                    "baseline_aggregate": result["baseline"],
                    "protocol_hash": self.config.protocol_hash,
                    "metric_hash": METRIC_HASH,
                    "seeds": seeds,
                    "notes": (
                        "automatic sealed opening after generation "
                        if automatic
                        else "explicit sealed opening; "
                    )
                    + f"{generation + 1}; aggregate-only boundary; not a generalization claim",
                }
            )
            self.ledger.record_event(
                self.run_id,
                "challenge_opened",
                payload={
                    "opening_index": opening_index,
                    "candidate": candidate.hash,
                    "delta": result.get("delta"),
                    "baseline": self.baseline_name,
                },
            )
        else:
            self.ledger.record_event(
                self.run_id,
                "challenge_failed",
                level="WARNING",
                payload={
                    "status": result["status"],
                    "error_code": result.get("error_code"),
                    "message": result.get("message"),
                },
            )
        return result

    def _run_challenge_mode(self) -> None:
        assert self.ledger is not None
        if not bool(self.config.get("challenge.enabled", False)):
            self.final_status = "challenge_disabled"
            self.ledger.record_event(
                self.run_id,
                "challenge_skipped",
                level="WARNING",
                payload={"reason": "challenge.enabled is false in the active configuration"},
            )
            return
        max_openings = int(self.config.get("challenge.max_openings_per_run", 1))
        if self.ledger.count_challenge_openings(self.run_id) >= max_openings:
            self.final_status = "exhausted"
            return
        candidate = self._resolve_candidate(self.requested_candidate)
        if candidate is None:
            self.final_status = "no_candidate"
            self.ledger.record_event(
                self.run_id,
                "challenge_skipped",
                level="WARNING",
                payload={
                    "reason": (
                        f"candidate {self.requested_candidate!r} not found in the recovered population"
                        if self.requested_candidate
                        else "no confirmed candidate in the recovered population"
                    )
                },
            )
            return
        result = self._open_challenge(self.generation, automatic=False, candidate_hash=candidate.hash)
        self.challenge_attempted = True
        self.final_status = "completed" if result.get("status") == "ok" else "challenge_failed"

    def _resolve_candidate(self, candidate_hash: str | None) -> Individual | None:
        assert self.population is not None
        if candidate_hash:
            for individual in self.population.all():
                if individual.hash == candidate_hash or individual.hash.startswith(candidate_hash):
                    return individual
            # Explicit challenge requests may name a candidate that has been
            # pruned from the live population. Recover it from the ledger.
            if self.ledger is not None:
                try:
                    row = self.ledger.get_candidate(self.run_id, candidate_hash)
                    if row is None and len(candidate_hash) >= 4:
                        matches = self.ledger.find_candidates(self.run_id, candidate_hash, limit=2)
                        row = matches[0] if len(matches) == 1 else None
                    if row is not None:
                        genome = Genome(row["genome"], **safety_kwargs(self.config))
                        individual = Individual(
                            genome=genome,
                            generation=int(row.get("generation", 0)),
                            operator=str(row.get("operator") or "ledger_recovery"),
                            parents=[str(row["parent_hash"])] if row.get("parent_hash") else [],
                        )
                        for trial in self.ledger.trials_for_candidate(self.run_id, genome.hash):
                            aggregate = trial.get("aggregate")
                            if trial.get("stage") == "confirm" and trial.get("status") == "ok" and isinstance(aggregate, dict):
                                individual.confirm_score = float(aggregate["mean"])
                                individual.confirm_aggregate = aggregate_summary(aggregate)
                                individual.status = "confirmed"
                                break
                        return individual
                except Exception:
                    return None
            return None
        if self.incumbent is not None:
            return self.incumbent
        confirmed = [
            individual for individual in self.population.all() if individual.confirm_score is not None
        ]
        if not confirmed:
            return None
        return max(confirmed, key=lambda item: (item.confirm_score or -1.0, -item.complexity))

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _cached_result(self, request: EvalRequest) -> dict[str, Any] | None:
        assert self.ledger is not None
        row = self.ledger.find_successful_trial(request.cache_key)
        if row is None:
            return None
        aggregate = row.get("aggregate")
        if not isinstance(aggregate, dict):
            return None
        if aggregate.get("metric_hash") != METRIC_HASH:
            return None
        if set(aggregate.get("per_task", {})) != set(request.task_ids):
            return None
        if int(aggregate.get("n_seeds", -1)) != len(request.seeds):
            return None
        return {
            "status": "ok",
            "aggregate": aggregate,
            "error_type": None,
            "error_code": None,
            "message": "",
            "elapsed_seconds": row.get("elapsed_seconds") or 0.0,
            "peak_rss_mb": row.get("peak_rss_mb"),
            "exit_code": row.get("exit_code"),
            "worker_pid": None,
            "memory_cap_enforced": True,
            "cache_key": request.cache_key,
            "cache_hit": True,
            "stage": request.stage,
            "task_ids": request.task_ids,
            "seeds": request.seeds,
        }

    def _finish_trial(self, result: dict[str, Any], trial_id: str, stage: str) -> None:
        assert self.ledger is not None
        assert self.artifacts is not None
        artifact_path = None
        artifact_sha = None
        if result["status"] == "ok" and result.get("aggregate") is not None:
            artifact = self.artifacts.put_json(result["aggregate"], kind=f"{stage}_aggregate")
            if artifact.stored:
                artifact_path = artifact.relative_path
                artifact_sha = artifact.sha256
        status = str(result["status"])
        error_type = result.get("error_type")
        if error_type is None and status in ("invalid", "timeout", "memory"):
            error_type = "candidate"
        self.ledger.finish_trial(
            trial_id=trial_id,
            status=status,
            aggregate=result.get("aggregate"),
            error_type=error_type,
            error_code=result.get("error_code"),
            message=result.get("message"),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha,
            elapsed_seconds=result.get("elapsed_seconds"),
            peak_rss_mb=result.get("peak_rss_mb"),
            exit_code=result.get("exit_code"),
        )

    def _record_archive(self, individual: Individual, info: dict[str, Any]) -> None:
        assert self.ledger is not None
        if info.get("pareto_added"):
            self.ledger.add_archive(
                run_id=self.run_id,
                kind="pareto",
                candidate_hash=individual.hash,
                reason="non-dominated score/complexity",
                payload={
                    "score": individual.rank_score,
                    "complexity": individual.complexity,
                    "level": individual.level,
                },
            )
        if info.get("new_cell") or info.get("cell_improved"):
            self.ledger.add_archive(
                run_id=self.run_id,
                kind="map_elite",
                candidate_hash=individual.hash,
                reason="new or improved behavior cell",
                payload={
                    "cell": info.get("cell"),
                    "score": individual.rank_score,
                    "novelty": info.get("novelty"),
                },
            )

    # ------------------------------------------------------------------
    # Persistence and finalization
    # ------------------------------------------------------------------
    def _checkpoint_payload(self) -> dict[str, Any]:
        assert self.population is not None
        assert self.archives is not None
        assert self.portfolio is not None
        return {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "mode": self.mode,
            "config_hash": self.config.config_hash,
            "protocol_hash": self.config.protocol_hash,
            "manifest_summary": self.manifest_summary,
            "started_at": self.started_at,
            "generation": self.generation,
            "trials_started": self.trials_started,
            "plateau": self.plateau,
            "promoted": self.promoted,
            "baselines_completed": self.baselines_completed,
            "challenge_attempted": self.challenge_attempted,
            "confirm_attempted": sorted(self.confirm_attempted),
            "elapsed_seconds": self._elapsed_seconds(),
            "population": self.population.to_dict(),
            "archives": self.archives.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "rng_state": _rng_state(self.rng),
            "incumbent": self.incumbent.to_dict() if self.incumbent is not None else None,
        }

    def _save_checkpoint(self, status: str) -> None:
        assert self.checkpoint is not None
        try:
            self.lock.heartbeat()
        except Exception:
            self.logger.warning("could not refresh the single-instance lock heartbeat", exc_info=True)
        info = self.checkpoint.save(self._checkpoint_payload(), generation=self.generation, status=status)
        self.trials_since_checkpoint = 0
        self.logger.info(
            "checkpoint saved at generation %s (%s)", info["generation"], info["checksum"][:12]
        )

    def _record_generation(self, stats: _GenerationStats, status: str) -> None:
        assert self.ledger is not None
        assert self.population is not None
        best_confirmed = self._best_confirmed()
        best = best_confirmed or self.population.best()
        self.ledger.save_generation(
            self.run_id,
            {
                "generation": stats.generation,
                "started_at": stats.started_at,
                "ended_at": utc_now(),
                "status": status,
                "proposals": stats.proposals,
                "valid": stats.valid,
                "invalid": stats.invalid,
                "confirmations": stats.confirmations,
                "promotions": stats.promotions,
                "best_hash": best.hash if best else None,
                "best_score": best.rank_score if best else None,
                "baseline_score": (
                    float(self.baseline["aggregate"]["mean"]) if self.baseline else None
                ),
                "plateau": self.plateau,
                "strategy": {"counts": stats.strategies, "meta_policy": self.meta.to_dict()},
                "summary": {
                    "quick_best": max(stats.quick_scores) if stats.quick_scores else None,
                    "confirm_best": max(stats.confirm_scores) if stats.confirm_scores else None,
                    "deadline_hit": stats.deadline_hit,
                    "challenge": stats.challenge,
                    "trials_started": self.trials_started,
                    "stagnation": analyze_stagnation(
                        plateau_generations=self.plateau,
                        patience=int(self.config.get("search.stagnation_patience_generations", 10)),
                        population=self.population.all(),
                        proposals=stats.proposals,
                        valid=stats.valid,
                        invalid=stats.invalid,
                        best_score=(best.rank_score if best is not None else None),
                    ),
                },
            },
        )
        self.logger.info(
            "generation %s %s: proposals=%s valid=%s invalid=%s confirms=%s promotions=%s",
            stats.generation,
            status,
            stats.proposals,
            stats.valid,
            stats.invalid,
            stats.confirmations,
            stats.promotions,
        )

    def _close(self) -> None:
        status = self.final_status or "failed"
        error = self.failure
        if self.ledger is not None:
            try:
                if error is not None:
                    atomic_write_json(
                        self.run_dir / "error.json",
                        {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc()[-12000:],
                            "at": utc_now(),
                        },
                    )
                self.ledger.set_run_status(
                    self.run_id,
                    status,
                    notes=(f"{type(error).__name__}: {error}" if error is not None else status)[:4000],
                )
                self.ledger.record_event(
                    self.run_id,
                    "run_finished",
                    level="ERROR" if status == "failed" else "INFO",
                    payload={"status": status, "generation": self.generation, "error": str(error) if error else None},
                )
            except Exception:
                self.logger.exception("failed to update final run status")
        if self.checkpoint is not None and self.population is not None:
            try:
                self.checkpoint.save(self._checkpoint_payload(), generation=self.generation, status=f"final_{status}")
            except Exception:
                self.logger.exception("failed to write final checkpoint")
        try:
            self._summary = self._build_summary()
            assert self._summary is not None
            atomic_write_json(self.run_dir / "summary.json", self._summary)
            self.heartbeat.write(
                self._heartbeat_payload({"status": status, "summary": self._summary}),
                force=True,
            )
        except Exception:
            self.logger.exception("failed to write run summary")
        try:
            if self.ledger is not None:
                self.ledger.close()
        finally:
            # On Windows an open FileHandler prevents temporary run directories
            # (and normal cleanup/archival tools) from deleting engine.log.
            self.engine._close_logging()

    def _build_summary(self) -> dict[str, Any]:
        population = self.population
        best_confirmed = self._best_confirmed() if population is not None else None
        best = best_confirmed or (population.best() if population is not None else None)
        baselines = []
        if self.ledger is not None:
            try:
                baselines = self.ledger.list_baselines(self.run_id)
            except Exception:
                baselines = []
        challenge_rows = []
        if self.ledger is not None:
            try:
                for row in self.ledger.tabled_rows("challenge_runs", self.run_id, limit=20):
                    item = dict(row)
                    item["candidate_aggregate"] = _json_field(item, "candidate_aggregate_json")
                    item["baseline_aggregate"] = _json_field(item, "baseline_aggregate_json")
                    item["seeds"] = _json_field(item, "seeds_json")
                    item.pop("candidate_aggregate_json", None)
                    item.pop("baseline_aggregate_json", None)
                    item.pop("seeds_json", None)
                    challenge_rows.append(item)
            except Exception:
                challenge_rows = []
        return {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.final_status or "failed",
            "engine_version": str(self.config.get("engine_version")),
            "created_at": self.started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": self._elapsed_seconds(),
            "elapsed_human": human_duration(self._elapsed_seconds()),
            "config_hash": self.config.config_hash,
            "protocol_hash": self.config.protocol_hash,
            "manifest_summary": self.manifest_summary,
            "data_manifest_hash": self.manifest_summary.get("data_manifest_hash"),
            "challenge_cohort": self.cohort,
            "generations_completed": max(0, self.generation + 1),
            "trials_started": self.trials_started,
            "promoted": self.promoted,
            "best_candidate": _individual_summary(best),
            "best_confirmed": _individual_summary(best_confirmed),
            "incumbent": _individual_summary(self.incumbent),
            "baselines": [
                {
                    "name": item["name"],
                    "status": item["status"],
                    "mean": (
                        float(item["aggregate"]["mean"])
                        if isinstance(item.get("aggregate"), dict)
                        else None
                    ),
                    "worst_family": (
                        item["aggregate"].get("worst_family")
                        if isinstance(item.get("aggregate"), dict)
                        else None
                    ),
                    "worst_family_score": (
                        float(item["aggregate"].get("worst_family_score"))
                        if isinstance(item.get("aggregate"), dict)
                        and item["aggregate"].get("worst_family_score") is not None
                        else None
                    ),
                }
                for item in baselines
            ],
            "strongest_baseline": self.baseline_name,
            "population": {
                "members": len(population.all()) if population is not None else 0,
                "pareto_entries": len(self.archives.pareto.entries) if self.archives else 0,
                "novelty_cells": len(self.archives.novelty.cells) if self.archives else 0,
            },
            "challenge_openings": challenge_rows,
            "arc": self.arc_profile,
            "hardware": self.engine.hardware.to_dict(),
            "resource_plan": asdict(self.engine.plan),
            "limitations": LIMITATIONS,
            "error": (
                {"type": type(self.failure).__name__, "message": str(self.failure)}
                if self.failure is not None
                else None
            ),
            "run_dir": str(self.run_dir),
            "ledger_path": str(self.run_dir / "ledger.sqlite3"),
        }

    def _best_confirmed(self) -> Individual | None:
        assert self.population is not None
        confirmed = [item for item in self.population.all() if item.confirm_score is not None]
        if not confirmed:
            return None
        return max(confirmed, key=lambda item: (item.confirm_score or -1.0, -item.complexity))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _specs_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        wanted = set(ids)
        specs = [spec for spec in self.config.dev_task_specs() if str(spec["task_id"]) in wanted]
        if set(str(spec["task_id"]) for spec in specs) != wanted:
            raise ConfigError(f"quick task ids are missing from open dev task specs: {sorted(wanted)}")
        return specs

    def _available_levels(self, generation: int) -> list[str]:
        configured = [str(item) for item in self.config.get("search.levels", ["A", "B", "C", "D"])]
        opening = dict(self.config.get("search.open_level_after_generation", {}))
        available = [level for level in configured if generation >= int(opening.get(level, 0))]
        return available or [configured[0]]

    def _deadline_reached(self) -> bool:
        return bool(self.deadline is not None and time.monotonic() >= self.deadline)

    def stop_requested(self) -> bool:
        if self.stop.requested():
            return True
        try:
            self.lock.heartbeat()
        except Exception:
            pass
        if self.extra_stop_callback is not None:
            try:
                return bool(self.extra_stop_callback())
            except Exception:
                return False
        return False

    def _elapsed_seconds(self) -> float:
        return self.prior_elapsed + max(0.0, time.monotonic() - self.session_started)

    def _check_run_storage_budget(self, *, force: bool = False) -> None:
        assert self.storage_guard is not None
        result = self.storage_guard.check(force=force)
        if not result["over_budget"]:
            return
        used_gb = int(result["bytes"]) / (1024**3)
        pause_gb = int(result["pause_at_bytes"]) / (1024**3)
        max_gb = int(result["max_bytes"]) / (1024**3)
        reason = (
            f"run storage budget reached: footprint {used_gb:.2f} GiB is at/above "
            f"the safe-pause threshold {pause_gb:.2f} GiB (configured cap {max_gb:.2f} GiB); "
            "AMDE paused after a durable checkpoint. Archive or remove unrelated data, "
            "then resume the same run after reviewing disk usage."
        )
        if self.logger:
            self.logger.error(reason)
        if self.ledger is not None:
            try:
                self.ledger.record_event(
                    self.run_id, "storage_budget_pause", level="WARNING",
                    payload={**result, "reason": reason},
                )
            except Exception:
                if self.logger:
                    self.logger.exception("could not record storage budget pause")
        raise ResourcePressure(reason)

    def _check_resources(self) -> None:
        # Honour the storage scanner interval during trial-level resource checks;
        # force a complete walk at startup and generation/checkpoint boundaries.
        self._check_run_storage_budget(force=False)
        assert self.sampler is not None
        assert self.ledger is not None
        sample = self.sampler.snapshot(force=True)
        self.ledger.record_resource_sample(self.run_id, sample)
        pressure = self.sampler.pressure(sample)
        if not pressure:
            return
        self.logger.warning("resource pressure: %s", "; ".join(pressure))
        self.ledger.record_event(
            self.run_id, "resource_pressure", level="WARNING", payload={"reasons": pressure, "sample": sample}
        )
        wait_limit = float(self.engine.policy.data["queue"].get("max_wait_seconds", 60))
        poll = max(0.5, float(self.engine.policy.data["queue"].get("poll_seconds", 2)))
        waited = 0.0
        while waited < wait_limit:
            if self.stop_requested():
                raise StopRequested("stop requested while waiting for resources")
            if self._deadline_reached():
                raise ResourcePressure("resource pressure persisted until the time budget ended")
            time.sleep(min(poll, max(0.0, wait_limit - waited)))
            waited += poll
            sample = self.sampler.snapshot(force=True)
            pressure = self.sampler.pressure(sample)
            if not pressure:
                self.ledger.record_event(
                    self.run_id, "resource_pressure_cleared", payload={"sample": sample}
                )
                return
        raise ResourcePressure("; ".join(pressure))

    def _write_run_manifest(self) -> None:
        assert self.ledger is not None
        manifest = {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "mode": self.mode,
            "created_at": self.started_at,
            "engine_version": str(self.config.get("engine_version")),
            "config_hash": self.config.config_hash,
            "protocol_hash": self.config.protocol_hash,
            "manifest_summary": self.manifest_summary,
            "config": self.config.effective_copy(),
            "hardware": self.engine.hardware.to_dict(),
            "resource_plan": asdict(self.engine.plan),
            "resource_policy": copy.deepcopy(self.engine.policy.data),
            "arc": self.arc_profile,
            "limitations": LIMITATIONS,
        }
        atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        self.ledger.record_event(
            self.run_id,
            "protocol_manifest",
            payload={
                "manifest_summary": self.manifest_summary,
                "arc_status": self.arc_profile.get("profile", {}).get("status"),
            },
        )


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        policy: ResourcePolicy | None = None,
        hardware: HardwareInfo | None = None,
        plan: ResourcePlan | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or ResourcePolicy.load(config.project_root)
        # Avoid nested BLAS/OpenMP oversubscription when the evaluator fans out
        # multiple isolated worker processes. Each process gets the same limit.
        set_blas_threads(int(self.policy.data.get("blas_threads", 1)))
        if hardware is not None:
            self.hardware = hardware
        else:
            requested_device = str(config.get("runtime.device", "auto"))
            gpu_policy_enabled = bool(self.policy.data.get("gpu", {}).get("enabled", False))
            self.hardware = detect_hardware(
                config.project_root,
                probe_torch=(gpu_policy_enabled and requested_device != "cpu"),
            )
        self.plan = plan or self.policy.compute_plan(
            self.hardware, str(config.get("runtime.device", "auto"))
        )
        # Effective backend is part of both the run identity and protocol hash.
        # Resuming a CPU checkpoint as CUDA (or vice versa) must fail closed.
        effective = str(self.plan.device_effective)
        if self.config.get("runtime.effective_device") != effective:
            self.config.set("runtime.effective_device", effective)
            self.config.refresh_hashes()
        self._logger: logging.Logger | None = None

    def _configure_logging(self, run_dir: Path) -> logging.Logger:
        logger = logging.getLogger(f"amde.{id(self)}")
        logger.setLevel(getattr(logging, str(self.config.get("logging.level", "INFO")).upper(), logging.INFO))
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        log_max_bytes = max(256 * 1024, int(self.config.get("logging.max_bytes", 5 * 1024 * 1024)))
        log_backups = max(1, min(20, int(self.config.get("logging.backup_count", 5))))
        file_handler = RotatingFileHandler(
            run_dir / "engine.log",
            maxBytes=log_max_bytes,
            backupCount=log_backups,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
        self._logger = logger
        return logger

    def _close_logging(self) -> None:
        logger = self._logger
        if logger is None:
            return
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
        self._logger = None

    def run(
        self,
        *,
        mode: str = "once",
        run_id: str | None = None,
        generations: int | None = None,
        max_seconds: float | None = None,
        max_trials: int | None = None,
        resume: bool = True,
        stop_callback: Callable[[], bool] | None = None,
        candidate_hash: str | None = None,
        isolate_stop: bool = False,
    ) -> dict[str, Any]:
        run_id = sanitize_id(run_id or new_run_id(mode))
        if mode == "smoke":
            if generations is None:
                generations = int(self.config.get("budgets.smoke_max_generations", 1))
            if max_seconds is None:
                max_seconds = float(self.config.get("budgets.smoke_max_seconds", 75))
        elif mode == "once":
            if generations is None:
                generations = int(self.config.get("budgets.default_generations", 3))
        elif mode == "continuous":
            if generations is None:
                generations = 0
        elif mode == "challenge":
            generations = 0
        else:
            raise ConfigError(f"unknown run mode: {mode}")
        if max_trials is None:
            max_trials = int(self.config.get("budgets.max_trials", 0))
        session = _RunSession(
            self,
            run_id=run_id,
            mode=mode,
            total_generations=int(generations or 0),
            max_seconds=float(max_seconds or 0.0),
            max_trials=int(max_trials or 0),
            resume=resume,
            stop_callback=stop_callback,
            candidate_hash=candidate_hash,
            isolate_stop=isolate_stop,
        )
        return session.execute()


# ----------------------------------------------------------------------
# Read-only operational helpers used by the CLI
# ----------------------------------------------------------------------
def _runs_dir(config: Config) -> Path:
    path = config.path("runs_dir")
    ensure_dir(path)
    return path


def latest_run_id(config: Config) -> str | None:
    runs = _runs_dir(config)
    candidates = []
    for path in runs.glob("*/ledger.sqlite3"):
        try:
            candidates.append((path.stat().st_mtime, path.parent.name))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def run_status(config: Config, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or latest_run_id(config)
    if not run_id:
        return {"status": "no_runs", "runs_dir": str(config.path("runs_dir"))}
    run_id = sanitize_id(run_id)
    run_dir = config.run_dir(run_id)
    ledger_path = run_dir / "ledger.sqlite3"
    if not ledger_path.exists():
        raise ConfigError(f"run not found: {run_id}")
    with Ledger(ledger_path) as ledger:
        run = ledger.get_run(run_id)
        if run is None:
            raise ConfigError(f"run {run_id} has no ledger row")
        baselines = ledger.list_baselines(run_id)
        generations = []
        for row in ledger.tabled_rows("generations", run_id, limit=500):
            item = dict(row)
            item["strategy"] = _json_field(item, "strategy_json")
            item["summary"] = _json_field(item, "summary_json")
            item.pop("strategy_json", None)
            item.pop("summary_json", None)
            generations.append(item)
        generations.sort(key=lambda item: int(item["generation"]))
        decisions = ledger.tabled_rows("decisions", run_id, limit=200)
        best = ledger.best_dev_candidate(run_id)
        latest = ledger.latest_challenge_opening(run_id)
        trial_counts = ledger.trial_status_counts(run_id)
        stage_counts = {
            stage: ledger.trial_status_counts(run_id, stage=stage)
            for stage in ("quick", "confirm")
        }
        events = ledger.latest_events(run_id, limit=20)
    heartbeat = HeartbeatWriter(run_dir, run_id=run_id).read()
    lock_path = config.path("runs_dir") / ".amde.lock"
    lock_data = None
    try:
        lock_data = read_json(lock_path)
    except Exception:
        lock_data = None
    lock_alive = bool(
        isinstance(lock_data, dict)
        and lock_data.get("run_id") == run_id
        and default_pid_alive(int(lock_data.get("pid", -1)))
    )
    checkpoint = CheckpointManager(run_dir)
    stop = StopController(
        run_dir,
        stop_name=str(config.get("paths.stop_file_name", "STOP")),
        global_runs_dir=config.path("runs_dir"),
    )
    summary_path = run_dir / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
        except Exception:
            summary = None
    status = str(run.get("status", "unknown"))
    if status == "running" and not lock_alive:
        status = "stale_running"
    return {
        "run_id": run_id,
        "run": run,
        "status": status,
        "run_dir": str(run_dir),
        "heartbeat": heartbeat,
        "lock": {"data": lock_data, "alive": lock_alive},
        "stop_requested": stop.requested(),
        "checkpoints": checkpoint.list_checksums(),
        "baselines": [
            {
                "name": item["name"],
                "status": item["status"],
                "mean": (
                    float(item["aggregate"]["mean"]) if isinstance(item.get("aggregate"), dict) else None
                ),
                "worst_family": (
                    item["aggregate"].get("worst_family")
                    if isinstance(item.get("aggregate"), dict)
                    else None
                ),
                "seeds": item.get("seeds"),
            }
            for item in baselines
        ],
        "generations": generations,
        "trial_counts": trial_counts,
        "stage_counts": stage_counts,
        "decisions": decisions,
        "best_dev": best,
        "latest_challenge": latest,
        "events": events,
        "summary": summary,
    }


def stop_run(config: Config, run_id: str | None = None, *, global_stop: bool = False) -> dict[str, Any]:
    run_id = run_id or latest_run_id(config)
    if not run_id:
        raise ConfigError("no run exists to stop")
    run_id = sanitize_id(run_id)
    run_dir = config.run_dir(run_id)
    if not run_dir.exists():
        raise ConfigError(f"run not found: {run_id}")
    controller = StopController(
        run_dir,
        stop_name=str(config.get("paths.stop_file_name", "STOP")),
        global_runs_dir=config.path("runs_dir"),
    )
    path = controller.request("user requested stop", global_stop=global_stop)
    return {"run_id": run_id, "stop_file": str(path), "global_stop": bool(global_stop)}


def generate_report(config: Config, run_id: str | None = None) -> Path:
    status = run_status(config, run_id)
    if status.get("status") == "no_runs":
        raise ConfigError("no runs are available to report")
    run_id = str(status["run_id"])
    run_dir = config.run_dir(run_id)
    summary = status.get("summary") or {}
    lines: list[str] = []
    lines.append(f"# AMDE Run Report: `{run_id}`")
    lines.append("")
    lines.append(f"- Status: **{status['status']}**")
    lines.append(f"- Mode: `{status.get('run', {}).get('mode')}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Config hash: `{status.get('run', {}).get('config_hash')}`")
    lines.append(f"- Protocol hash: `{status.get('run', {}).get('protocol_hash')}`")
    lines.append(f"- Manifest hash: `{status.get('run', {}).get('manifest_hash')}`")
    if summary:
        lines.append(f"- Elapsed: {summary.get('elapsed_human')}")
        lines.append(f"- Generations completed: {summary.get('generations_completed')}")
        lines.append(f"- Trials started: {summary.get('trials_started')}")
        lines.append(f"- Promoted candidate: {summary.get('promoted')}")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Baseline | Status | Mean dev score | Worst family |")
    lines.append("|---|---|---:|---|")
    for item in status.get("baselines", []):
        mean = item.get("mean")
        lines.append(
            f"| `{item['name']}` | {item['status']} | "
            f"{'—' if mean is None else f'{mean:.4f}'} | {item.get('worst_family') or '—'} |"
        )
    lines.append("")
    lines.append("## Generations")
    lines.append("")
    lines.append("| Gen | Status | Proposals | Valid | Confirms | Promotions | Best score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in status.get("generations", []):
        best = row.get("best_score")
        lines.append(
            f"| {row['generation']} | {row['status']} | {row.get('proposals', 0)} | "
            f"{row.get('valid', 0)} | {row.get('confirmations', 0)} | {row.get('promotions', 0)} | "
            f"{'—' if best is None else f'{float(best):.4f}'} |"
        )
    lines.append("")
    best = summary.get("best_confirmed") or summary.get("best_candidate")
    lines.append("## Best Candidate")
    lines.append("")
    if best:
        lines.append(f"- Hash: `{best.get('candidate_hash')}`")
        lines.append(f"- Level: `{best.get('level')}`")
        lines.append(f"- Quick score: {best.get('quick_score')}")
        lines.append(f"- Confirm score: {best.get('confirm_score')}")
        lines.append(f"- Complexity: {best.get('complexity')}")
        genome = best.get("genome") or {}
        lines.append(
            f"- Model: `{genome.get('model', {}).get('kind')}` hidden={genome.get('model', {}).get('hidden')}"
        )
        lines.append(
            f"- Optimizer: `{genome.get('optimizer', {}).get('kind')}` "
            f"lr={genome.get('recipe', {}).get('learning_rate')} "
            f"epochs={genome.get('recipe', {}).get('epochs')}"
        )
    else:
        lines.append("No candidate has been evaluated.")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    decisions = status.get("decisions", [])
    if decisions:
        lines.append("| Gen | Candidate | Decision | Reason |")
        lines.append("|---:|---|---|---|")
        for row in reversed(decisions[:30]):
            candidate = str(row.get("candidate_hash", ""))[:12]
            reason = str(row.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {row.get('generation')} | `{candidate}` | {row.get('decision')} | {reason[:240]} |")
    else:
        lines.append("No confirmations yet.")
    lines.append("")
    generations = status.get("generations", [])
    latest_stagnation = (
        (generations[-1].get("summary") or {}).get("stagnation")
        if generations and isinstance(generations[-1].get("summary"), dict)
        else None
    )
    if isinstance(latest_stagnation, dict):
        lines.append("## Search Diagnostics & Stagnation")
        lines.append("")
        lines.append(f"- Stagnated: `{latest_stagnation.get('is_stagnated')}`")
        lines.append(
            f"- Plateau generations: {latest_stagnation.get('plateau_generations')} / "
            f"patience {latest_stagnation.get('patience_generations')}"
        )
        lines.append(
            f"- Phenotype diversity: {latest_stagnation.get('phenotype_diversity')} "
            f"({latest_stagnation.get('unique_phenotypes')} unique / {latest_stagnation.get('population_size')} members)"
        )
        lines.append(f"- Invalid candidate rate: {latest_stagnation.get('invalid_rate')}")
        lines.append(f"- Automatic intervention: `{latest_stagnation.get('automatic_intervention')}`")
        for rec in latest_stagnation.get("recommendations") or []:
            lines.append(f"- Recommendation: {rec}")
        lines.append("")
    lines.append("## Challenge")
    lines.append("")
    challenge = status.get("latest_challenge")
    if challenge:
        lines.append(
            f"Latest sealed opening `{challenge.get('opening_index')}` on cohort "
            f"`{challenge.get('cohort')}`."
        )
    else:
        lines.append("No sealed challenge opening was recorded.")
    lines.append("")
    lines.append("## Interpretation Limits")
    lines.append("")
    for limitation in LIMITATIONS:
        lines.append(f"- {limitation}")
    if summary.get("arc"):
        lines.append(f"- ARC profile: {summary['arc'].get('profile', {}).get('status')} — "
                     f"{summary['arc'].get('profile', {}).get('reason')}")
    lines.append("")
    reports_dir = ensure_dir(config.reports_dir())
    target = reports_dir / f"{run_id}_report.md"
    atomic_write_text(target, "\n".join(lines) + "\n")
    return target


def engine_doctor(config: Config, *, probe_torch: bool = False) -> dict[str, Any]:
    policy = ResourcePolicy.load(config.project_root)
    hardware = detect_hardware(config.project_root, probe_torch=probe_torch)
    plan = policy.compute_plan(hardware, str(config.get("runtime.device", "auto")))
    return {
        "engine_version": str(config.get("engine_version")),
        "project_root": str(config.project_root),
        "config_hash": config.config_hash,
        "protocol_hash": config.protocol_hash,
        "config_path": str(config.source_path) if config.source_path else None,
        "hardware": hardware.to_dict(),
        "resource_policy": copy.deepcopy(policy.data),
        "policy_warnings": list(policy.warnings),
        "resource_plan": asdict(plan),
        "arc": arc_status(config.arc_path()),
        "limitations": LIMITATIONS,
    }

"""Tests for default skill lifecycle hooks (DS-9, DS-10, DS-11)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from framework.graph.event_loop.event_publishing import run_hooks
from framework.graph.event_loop.types import HookContext, HookResult
from framework.graph.executor import _run_phase_transition_hooks
from framework.graph.node import DataBuffer
from framework.skills.config import DefaultSkillConfig, SkillsConfig
from framework.skills.defaults import DATA_BUFFER_KEYS, SKILL_REGISTRY, DefaultSkillManager
from framework.skills.manager import SkillsManager


class TestHookContextExtension:
    """HookContext supports new optional fields for lifecycle hooks."""

    def test_new_fields_default_to_none(self):
        ctx = HookContext(event="session_start", trigger=None, system_prompt="test")
        assert ctx.shared_memory is None
        assert ctx.iteration is None
        assert ctx.node_id is None

    def test_new_fields_populated(self):
        memory = MagicMock()
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=5,
            node_id="worker-1",
        )
        assert ctx.shared_memory is memory
        assert ctx.iteration == 5
        assert ctx.node_id == "worker-1"


class TestRunHooksExtended:
    """run_hooks() passes new HookContext fields to callbacks."""

    def _make_conversation(self):
        conv = MagicMock()
        conv.system_prompt = "test prompt"
        conv.update_system_prompt = MagicMock()
        conv.add_user_message = AsyncMock()
        return conv

    def test_hooks_receive_shared_memory_and_iteration(self):
        received = {}

        async def capture_hook(ctx: HookContext) -> HookResult | None:
            received["shared_memory"] = ctx.shared_memory
            received["iteration"] = ctx.iteration
            received["node_id"] = ctx.node_id
            return None

        memory = MagicMock()
        hooks = {"iteration_boundary": [capture_hook]}
        asyncio.get_event_loop().run_until_complete(
            run_hooks(
                hooks,
                "iteration_boundary",
                self._make_conversation(),
                shared_memory=memory,
                iteration=3,
                node_id="worker-1",
            )
        )
        assert received["shared_memory"] is memory
        assert received["iteration"] == 3
        assert received["node_id"] == "worker-1"

    def test_hooks_without_new_fields_still_work(self):
        """Existing callers passing no new fields still work (backward compat)."""
        called = []

        async def simple_hook(ctx: HookContext) -> HookResult | None:
            called.append(ctx.event)
            assert ctx.shared_memory is None
            assert ctx.iteration is None
            assert ctx.node_id is None
            return None

        hooks = {"session_start": [simple_hook]}
        asyncio.get_event_loop().run_until_complete(
            run_hooks(hooks, "session_start", self._make_conversation())
        )
        assert called == ["session_start"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(**skill_overrides: dict) -> DefaultSkillManager:
    """Create a loaded manager with optional per-skill config overrides."""
    configs: dict[str, DefaultSkillConfig] = {}
    for name, overrides in skill_overrides.items():
        configs[name] = DefaultSkillConfig.from_dict(overrides)
    config = SkillsConfig(default_skills=configs)
    mgr = DefaultSkillManager(config=config)
    mgr.load()
    return mgr


def _solo_manager(skill_name: str, overrides: dict | None = None) -> DefaultSkillManager:
    """Manager with only *skill_name* enabled; all others disabled."""
    all_configs: dict[str, dict] = {}

    for name in SKILL_REGISTRY:
        if name == skill_name:
            all_configs[name] = overrides or {}
        else:
            all_configs[name] = {"enabled": False}
    return _make_manager(**all_configs)


# ---------------------------------------------------------------------------
# register_hooks structure tests
# ---------------------------------------------------------------------------


class TestBatchSkippedKey:
    def test_batch_skipped_in_shared_memory_keys(self):
        assert "_batch_skipped" in DATA_BUFFER_KEYS


class TestRegisterHooks:
    """DefaultSkillManager.register_hooks() registers per-skill callbacks."""

    def test_registers_iteration_boundary_hooks(self):
        mgr = _make_manager()
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        # quality-monitor, note-taking, batch-ledger → 3 iteration hooks
        assert len(hooks.get("iteration_boundary", [])) == 3

    def test_registers_node_complete_hooks(self):
        mgr = _make_manager()
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        # note-taking, batch-ledger, context-preservation → 3 complete hooks
        assert len(hooks.get("node_complete", [])) == 3

    def test_registers_phase_transition_hooks(self):
        mgr = _make_manager()
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        # context-preservation → 1 transition hook
        assert len(hooks.get("phase_transition", [])) == 1

    def test_disabled_skill_no_hooks(self):
        mgr = _make_manager(**{"hive.quality-monitor": {"enabled": False}})
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        # Only note-taking + batch-ledger → 2 iteration hooks
        assert len(hooks.get("iteration_boundary", [])) == 2

    def test_all_disabled_no_hooks(self):
        config = SkillsConfig(all_defaults_disabled=True)
        mgr = DefaultSkillManager(config=config)
        mgr.load()
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        assert hooks == {}


# ---------------------------------------------------------------------------
# Per-skill callback tests
# ---------------------------------------------------------------------------


class TestQualityMonitorHook:
    """hive.quality-monitor iteration_boundary fires every N iterations."""

    def _get_hooks(self, assessment_interval: int = 5) -> list:
        mgr = _solo_manager("hive.quality-monitor", {"assessment_interval": assessment_interval})
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        return hooks.get("iteration_boundary", [])

    def test_fires_at_interval(self):
        hooks = self._get_hooks(assessment_interval=3)
        hook = hooks[0]
        results = []
        for i in range(9):
            ctx = HookContext(
                event="iteration_boundary",
                trigger=None,
                system_prompt="test",
                iteration=i,
            )
            result = asyncio.get_event_loop().run_until_complete(hook(ctx))
            results.append(result)
        # Fires at iterations 2, 5, 8 (0-indexed: (i+1) % 3 == 0)
        assert results[2] is not None
        assert results[5] is not None
        assert results[8] is not None
        assert results[0] is None
        assert results[1] is None
        assert results[3] is None

    def test_uses_default_assessment_interval_from_skill_defaults(self):
        """When no override is set, _SKILL_DEFAULTS assessment_interval=5 is used."""
        mgr = _solo_manager("hive.quality-monitor")  # no overrides
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        hook = hooks["iteration_boundary"][0]
        # iteration 3 → (3+1)%5 != 0 → no fire
        ctx = HookContext(event="iteration_boundary", trigger=None, system_prompt="t", iteration=3)
        assert asyncio.get_event_loop().run_until_complete(hook(ctx)) is None
        # iteration 4 → (4+1)%5 == 0 → fires
        ctx = HookContext(event="iteration_boundary", trigger=None, system_prompt="t", iteration=4)
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is not None
        assert "QUALITY CHECK" in result.inject

    def test_inject_contains_quality_check(self):
        hooks = self._get_hooks(assessment_interval=1)
        hook = hooks[0]
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            iteration=0,
        )
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is not None
        assert "QUALITY CHECK" in result.inject
        assert "_quality_log" in result.inject


class TestNoteTakingHook:
    """hive.note-taking hooks: staleness detection + final snapshot."""

    def _get_hooks(self, staleness_threshold: int = 5) -> tuple[list, list]:
        mgr = _solo_manager("hive.note-taking", {"staleness_threshold": staleness_threshold})
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        return hooks.get("iteration_boundary", []), hooks.get("node_complete", [])

    def test_no_warning_when_notes_not_initialized(self):
        iter_hooks, _ = self._get_hooks(staleness_threshold=3)
        hook = iter_hooks[0]
        memory = DataBuffer()
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=10,
        )
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is None

    def test_warning_when_stale(self):
        iter_hooks, _ = self._get_hooks(staleness_threshold=3)
        hook = iter_hooks[0]
        memory = DataBuffer()
        memory.write("_notes_updated_at", 2, validate=False)
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=6,
        )
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is not None
        assert "NOTES STALE" in result.inject

    def test_no_warning_when_fresh(self):
        iter_hooks, _ = self._get_hooks(staleness_threshold=5)
        hook = iter_hooks[0]
        memory = DataBuffer()
        memory.write("_notes_updated_at", 8, validate=False)
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=10,
        )
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is None  # Only 2 iterations stale, threshold is 5

    def test_node_complete_logs_notes(self, caplog):
        _, complete_hooks = self._get_hooks()
        hook = complete_hooks[0]
        memory = DataBuffer()
        memory.write("_working_notes", "## Objective\nDo stuff", validate=False)
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.INFO):
            result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is None
        assert "hive.note-taking" in caplog.text
        assert "notes_length=" in caplog.text


class TestBatchLedgerHook:
    """hive.batch-ledger hooks: checkpoint + completeness check."""

    def _get_hooks(self, checkpoint_every_n: int = 10) -> tuple[list, list]:
        mgr = _solo_manager("hive.batch-ledger", {"checkpoint_every_n": checkpoint_every_n})
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        return hooks.get("iteration_boundary", []), hooks.get("node_complete", [])

    def test_checkpoint_skips_when_no_batch(self):
        iter_hooks, _ = self._get_hooks(checkpoint_every_n=1)
        hook = iter_hooks[0]
        memory = DataBuffer()
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=0,
        )
        result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is None

    def test_checkpoint_fires_at_interval(self, caplog):
        iter_hooks, _ = self._get_hooks(checkpoint_every_n=3)
        hook = iter_hooks[0]
        memory = DataBuffer()
        memory.write("_batch_total", 10, validate=False)
        memory.write("_batch_completed", 3, validate=False)
        # iteration 2 → (2+1)%3==0 → fires
        ctx = HookContext(
            event="iteration_boundary",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            iteration=2,
        )
        with caplog.at_level(logging.INFO):
            result = asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert result is None  # checkpoint only logs, no inject
        assert "checkpoint" in caplog.text

    def test_complete_warns_on_incomplete_batch(self, caplog):
        _, complete_hooks = self._get_hooks()
        hook = complete_hooks[0]
        memory = DataBuffer()
        memory.write("_batch_total", 10, validate=False)
        memory.write("_batch_completed", 5, validate=False)
        memory.write("_batch_failed", 1, validate=False)
        memory.write("_batch_skipped", 0, validate=False)
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "INCOMPLETE" in caplog.text
        assert "6/10" in caplog.text

    def test_complete_silent_when_all_processed(self, caplog):
        _, complete_hooks = self._get_hooks()
        hook = complete_hooks[0]
        memory = DataBuffer()
        memory.write("_batch_total", 10, validate=False)
        memory.write("_batch_completed", 8, validate=False)
        memory.write("_batch_failed", 1, validate=False)
        memory.write("_batch_skipped", 1, validate=False)
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "INCOMPLETE" not in caplog.text


class TestContextPreservationHook:
    """hive.context-preservation: handoff verification + phase transition logging."""

    def _get_hooks(self, require_handoff: bool = True) -> tuple[list, list]:
        mgr = _solo_manager("hive.context-preservation", {"require_handoff": require_handoff})
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        return hooks.get("node_complete", []), hooks.get("phase_transition", [])

    def test_default_require_handoff_is_true(self, caplog):
        """With no override, require_handoff defaults to True per PRD."""
        mgr = _solo_manager("hive.context-preservation")  # no overrides
        hooks: dict[str, list] = {}
        mgr.register_hooks(hooks)
        hook = hooks["node_complete"][0]
        memory = DataBuffer()
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "MISSING_HANDOFF" in caplog.text

    def test_warns_on_missing_handoff(self, caplog):
        complete_hooks, _ = self._get_hooks(require_handoff=True)
        hook = complete_hooks[0]
        memory = DataBuffer()
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "MISSING_HANDOFF" in caplog.text

    def test_no_warning_when_handoff_present(self, caplog):
        complete_hooks, _ = self._get_hooks(require_handoff=True)
        hook = complete_hooks[0]
        memory = DataBuffer()
        memory.write("_handoff_context", "Summary of work done", validate=False)
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "MISSING_HANDOFF" not in caplog.text

    def test_no_warning_when_require_handoff_disabled(self, caplog):
        complete_hooks, _ = self._get_hooks(require_handoff=False)
        hook = complete_hooks[0]
        memory = DataBuffer()
        ctx = HookContext(
            event="node_complete",
            trigger=None,
            system_prompt="test",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "MISSING_HANDOFF" not in caplog.text

    def test_phase_transition_logs_handoff(self, caplog):
        _, transition_hooks = self._get_hooks()
        hook = transition_hooks[0]
        memory = DataBuffer()
        memory.write("_handoff_context", "Next node needs this data", validate=False)
        ctx = HookContext(
            event="phase_transition",
            trigger=None,
            system_prompt="",
            shared_memory=memory,
            node_id="worker-1",
        )
        with caplog.at_level(logging.INFO):
            asyncio.get_event_loop().run_until_complete(hook(ctx))
        assert "handoff_available" in caplog.text


class TestHookErrorResilience:
    """Hook errors are logged but never block execution."""

    def test_broken_hook_does_not_block_others(self):
        called = []

        async def broken_hook(ctx: HookContext) -> HookResult | None:
            raise RuntimeError("intentional test error")

        async def good_hook(ctx: HookContext) -> HookResult | None:
            called.append(True)
            return None

        hooks = {"iteration_boundary": [broken_hook, good_hook]}
        conv = MagicMock()
        conv.system_prompt = "test"
        asyncio.get_event_loop().run_until_complete(run_hooks(hooks, "iteration_boundary", conv))
        assert called == [True]


# ---------------------------------------------------------------------------
# SkillsManager facade tests
# ---------------------------------------------------------------------------


class TestSkillsManagerLifecycleHooks:
    """SkillsManager.lifecycle_hooks property delegates to DefaultSkillManager."""

    def test_lifecycle_hooks_populated(self):

        mgr = SkillsManager()
        mgr.load()
        hooks = mgr.lifecycle_hooks
        assert "iteration_boundary" in hooks
        assert "node_complete" in hooks
        assert "phase_transition" in hooks

    def test_lifecycle_hooks_empty_when_not_loaded(self):

        mgr = SkillsManager()
        # Don't call load()
        hooks = mgr.lifecycle_hooks
        assert hooks == {}


# ---------------------------------------------------------------------------
# Phase transition hook runner tests
# ---------------------------------------------------------------------------


class TestPhaseTransitionFanOut:
    """Phase transition hooks fire on fan-out edge traversals (executor integration)."""

    @pytest.mark.asyncio
    async def test_phase_transition_fires_for_each_fanout_edge(self):
        from unittest.mock import MagicMock as _Mock

        from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
        from framework.graph.executor import GraphExecutor
        from framework.graph.goal import Goal
        from framework.graph.node import NodeContext, NodeProtocol, NodeResult, NodeSpec
        from framework.runtime.core import Runtime

        class _SuccessNode(NodeProtocol):
            async def execute(self, ctx: NodeContext) -> NodeResult:
                return NodeResult(success=True, output={"done": True}, tokens_used=1, latency_ms=1)

        source = NodeSpec(
            id="src",
            name="Source",
            description="entry",
            node_type="event_loop",
            output_keys=["data"],
        )
        b1 = NodeSpec(
            id="b1",
            name="B1",
            description="branch 1",
            node_type="event_loop",
            output_keys=["b1_out"],
        )
        b2 = NodeSpec(
            id="b2",
            name="B2",
            description="branch 2",
            node_type="event_loop",
            output_keys=["b2_out"],
        )
        graph = GraphSpec(
            id="g",
            goal_id="g1",
            name="Fanout",
            entry_node="src",
            nodes=[source, b1, b2],
            edges=[
                EdgeSpec(id="e1", source="src", target="b1", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e2", source="src", target="b2", condition=EdgeCondition.ON_SUCCESS),
            ],
            terminal_nodes=["b1", "b2"],
        )

        # Track phase_transition hook calls
        transition_calls: list[str] = []

        async def _track(ctx: HookContext) -> HookResult | None:
            transition_calls.append(ctx.node_id or "?")
            return None

        rt = _Mock(spec=Runtime)
        rt.start_run = _Mock(return_value="run_id")
        rt.decide = _Mock(return_value="d")
        rt.record_outcome = _Mock()
        rt.end_run = _Mock()
        rt.report_problem = _Mock()
        rt.set_node = _Mock()

        executor = GraphExecutor(
            runtime=rt,
            enable_parallel_execution=True,
            lifecycle_hooks={"phase_transition": [_track]},
        )
        executor.register_node("src", _SuccessNode())
        executor.register_node("b1", _SuccessNode())
        executor.register_node("b2", _SuccessNode())

        result = await executor.execute(graph, Goal(id="g1", name="T", description="t"), {})

        assert result.success
        # Phase transition hooks should fire for both fan-out edges (src → b1, src → b2)
        assert len(transition_calls) == 2
        assert all(n == "src" for n in transition_calls)


class TestPhaseTransitionHookRunner:
    """Standalone phase transition hook runner for GraphExecutor."""

    def test_fires_phase_transition_hooks(self):

        called = []

        async def track_hook(ctx: HookContext) -> HookResult | None:
            called.append(
                {
                    "event": ctx.event,
                    "node_id": ctx.node_id,
                    "shared_memory": ctx.shared_memory,
                }
            )
            return None

        memory = DataBuffer()
        hooks = {"phase_transition": [track_hook]}
        asyncio.get_event_loop().run_until_complete(
            _run_phase_transition_hooks(hooks, memory, "node-a", "node-b")
        )
        assert len(called) == 1
        assert called[0]["event"] == "phase_transition"
        assert called[0]["node_id"] == "node-a"
        assert called[0]["shared_memory"] is memory

    def test_error_in_hook_does_not_propagate(self):

        async def broken_hook(ctx: HookContext) -> HookResult | None:
            raise RuntimeError("boom")

        hooks = {"phase_transition": [broken_hook]}
        memory = DataBuffer()
        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            _run_phase_transition_hooks(hooks, memory, "a", "b")
        )

    def test_no_hooks_is_noop(self):

        memory = DataBuffer()
        asyncio.get_event_loop().run_until_complete(
            _run_phase_transition_hooks({}, memory, "a", "b")
        )

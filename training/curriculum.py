"""curriculum.py — QKTJ curriculum scheduling for QIG-Native training."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Canonical phase definitions (matches qig_kernel/regime.py)
CURRICULUM_PHASES: Dict[str, Dict] = {
    "phase0_identity": {
        "ordinal": 0,
        "phi_floor": 0.0,
        "phi_target": 0.3,
        "regime_idx": 0,
        "gcs_prefix": "qktj/phase0_identity/",
        "weight": 0.10,
        "description": "Identity formation — basin attractors stabilise at vocabulary level",
    },
    "phase1_coupling": {
        "ordinal": 1,
        "phi_floor": 0.25,
        "phi_target": 0.5,
        "regime_idx": 1,
        "gcs_prefix": "qktj/phase1_coupling/",
        "weight": 0.20,
        "description": "Coupling — cross-token basin correlations emerge",
    },
    "phase2_integration": {
        "ordinal": 2,
        "phi_floor": 0.45,
        "phi_target": 0.7,
        "regime_idx": 2,
        "gcs_prefix": "qktj/phase2_integration/",
        "weight": 0.40,
        "description": "Integration — sustained phi coherence across sequences",
    },
    "phase3_temporal": {
        "ordinal": 3,
        "phi_floor": 0.65,
        "phi_target": 0.9,
        "regime_idx": 3,
        "gcs_prefix": "qktj/phase3_temporal/",
        "weight": 0.30,
        "description": "Temporal — long-range basin geodesic consistency",
    },
}


@dataclass
class PhaseState:
    """Tracks current phase progression state."""
    current_phase: str = "phase0_identity"
    steps_in_phase: int = 0
    phi_history: List[float] = None
    phi_window: int = 100

    def __post_init__(self):
        if self.phi_history is None:
            self.phi_history = []


class CurriculumScheduler:
    """Manage curriculum phase advancement based on phi coherence metrics.

    Phase advance criteria:
    1. phi_history has at least phi_window observations
    2. min(phi_history) >= phi_floor for current phase
    3. mean(phi_history) >= phi_target for current phase

    Phase retreat criteria (soft penalty, not hard reset):
    - phi drops below phi_floor after being above phi_target
    - Increases decoherence gamma to encourage re-exploration
    """

    def __init__(
        self,
        phi_window: int = 100,
        start_phase: str = "phase0_identity",
    ):
        self._state = PhaseState(
            current_phase=start_phase,
            phi_window=phi_window,
        )
        self._phase_keys = list(CURRICULUM_PHASES.keys())
        self._advance_log: List[Tuple[int, str, str]] = []  # (step, from, to)
        self._global_step: int = 0

    @property
    def current_phase(self) -> str:
        return self._state.current_phase

    @property
    def current_phase_config(self) -> Dict:
        return CURRICULUM_PHASES[self._state.current_phase]

    @property
    def current_phase_ordinal(self) -> int:
        return self.current_phase_config["ordinal"]

    def step(self, phi: float, regime_idx: Optional[int] = None) -> Dict:
        """Record phi observation and check for phase transition.

        Returns:
            dict with 'advanced' (bool), 'new_phase' (str), 'reason' (str)
        """
        self._global_step += 1
        self._state.steps_in_phase += 1
        self._state.phi_history.append(phi)
        if len(self._state.phi_history) > self._state.phi_window:
            self._state.phi_history.pop(0)

        result = {"advanced": False, "new_phase": self._state.current_phase, "reason": ""}
        cfg = CURRICULUM_PHASES[self._state.current_phase]

        if len(self._state.phi_history) < self._state.phi_window:
            result["reason"] = f"warming_up_{len(self._state.phi_history)}/{self._state.phi_window}"
            return result

        min_phi = min(self._state.phi_history)
        avg_phi = sum(self._state.phi_history) / len(self._state.phi_history)

        if min_phi < cfg["phi_floor"]:
            result["reason"] = f"phi_floor_violation_min={min_phi:.3f}"
            return result

        if avg_phi >= cfg["phi_target"]:
            current_idx = self._phase_keys.index(self._state.current_phase)
            if current_idx < len(self._phase_keys) - 1:
                old_phase = self._state.current_phase
                new_phase = self._phase_keys[current_idx + 1]
                self._state.current_phase = new_phase
                self._state.steps_in_phase = 0
                self._state.phi_history.clear()
                self._advance_log.append((self._global_step, old_phase, new_phase))
                result["advanced"] = True
                result["new_phase"] = new_phase
                result["reason"] = f"phi_target_met_avg={avg_phi:.3f}"
            else:
                result["reason"] = "final_phase_maintained"
        else:
            result["reason"] = f"phi_below_target_avg={avg_phi:.3f}_target={cfg['phi_target']}"

        return result

    def get_loss_weights(self) -> Dict[str, float]:
        """Return curriculum-adjusted loss weights for current phase."""
        return {
            "basin_geodesic": 0.4,
            "regime_classification": 0.3,
            "phi_coherence": 0.2,
            "language": 0.1,
        }

    def get_gcs_prefix(self) -> str:
        """GCS path prefix for current phase data."""
        return self.current_phase_config["gcs_prefix"]

    def state_dict(self) -> Dict:
        return {
            "current_phase": self._state.current_phase,
            "steps_in_phase": self._state.steps_in_phase,
            "phi_history": self._state.phi_history,
            "phi_window": self._state.phi_window,
            "global_step": self._global_step,
            "advance_log": self._advance_log,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._state.current_phase = state["current_phase"]
        self._state.steps_in_phase = state["steps_in_phase"]
        self._state.phi_history = state["phi_history"]
        self._state.phi_window = state["phi_window"]
        self._global_step = state["global_step"]
        self._advance_log = state["advance_log"]

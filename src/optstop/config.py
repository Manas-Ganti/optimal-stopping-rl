"""Typed configuration. Everything numeric comes from configs/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


class ContractCfg(BaseModel):
    kind: Literal["put", "call"] = "put"
    strike: float
    expiry: float
    steps: int
    rate: float


class MarketCfg(BaseModel):
    spot: float
    vol: float


class BoundaryTrainingCfg(BaseModel):
    spot_low: float
    spot_high: float


class LSMCfg(BaseModel):
    basis: Literal["laguerre", "polynomial"] = "laguerre"
    degree: int = 3
    n_paths: int = 100_000


class ExtraTreesCfg(BaseModel):
    n_estimators: int = 50
    min_samples_leaf: int = 50


class FittedQCfg(BaseModel):
    regressor: Literal["rbf", "extra_trees"] = "rbf"
    n_paths: int = 20_000
    n_iterations: int | None = None
    rbf_centers_x: int = 24
    rbf_centers_tau: int = 12
    rbf_width: float = 1.0
    ridge: float = 1e-6
    rbf_normalized: bool = True
    control_variate: bool = True
    cv_centers_x: int = 8
    extra_trees: ExtraTreesCfg = ExtraTreesCfg()


class EvaluationCfg(BaseModel):
    n_paths: int = 200_000
    ci_level: float = 0.95


class BoundaryCfg(BaseModel):
    x_grid_min: float = 0.4
    x_grid_max: float = 1.6
    x_grid_points: int = 1201


class MertonCase(BaseModel):
    spot: float
    vol: float
    expiry: float
    intensity: float
    jump_mean: float
    jump_std: float


class GateACfg(BaseModel):
    n_paths: int
    z_max: float
    merton_cases: list[MertonCase]


class GateBCfg(BaseModel):
    abs_tol: float
    mean_abs_tol: float


class GateCCfg(BaseModel):
    max_early_exercise_rate: float
    z_max: float
    n_paths: int


class GateDCfg(BaseModel):
    mean_abs_boundary_tol: float
    max_abs_boundary_tol: float
    rel_value_tol: float


class GatesCfg(BaseModel):
    a: GateACfg
    b: GateBCfg
    c: GateCCfg
    d: GateDCfg


class MertonCfg(BaseModel):
    intensity: float
    jump_mean: float
    jump_std: float


class HestonCfg(BaseModel):
    kappa: float
    theta_scale: float
    xi: float
    rho: float


class MisspecCfg(BaseModel):
    n_paths: int
    merton: MertonCfg
    heston: HestonCfg


class RealCfg(BaseModel):
    tickers: list[str]
    start: str
    end: str
    rate: float
    trading_days_per_year: int = 252
    calibration_days: int = 252
    vol_grid: list[float]
    cache_dir: str = "data_cache"


class Config(BaseModel):
    seed: int
    contract: ContractCfg
    market: MarketCfg
    boundary_training: BoundaryTrainingCfg
    lsm: LSMCfg
    fitted_q: FittedQCfg
    evaluation: EvaluationCfg
    boundary: BoundaryCfg
    gates: GatesCfg
    misspecification: MisspecCfg
    real: RealCfg


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG) as f:
        return Config.model_validate(yaml.safe_load(f))

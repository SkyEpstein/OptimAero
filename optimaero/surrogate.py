"""The surrogate interface — the single plug-in point for the ML.

Everything downstream (physics coupling, inverse-design optimizer, CAD export) is written
against `Surrogate`, never against a concrete model. The Phase-2 bake-off produces a trained
`Surrogate` (predictor + confidence model); until then `NeuralFoilSurrogate` is a working
stand-in so the whole product can be built and tested end-to-end.

A `Surrogate` returns, for a geometry at a condition, the three section coefficients AND the
confidence model's view of its own error — the trust gate that lets the system defer to CFD.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AeroPrediction:
    """One surrogate evaluation, with the confidence model's self-assessment."""
    Cl: float
    Cd: float
    Cm: float
    # Confidence model: predicted absolute error on each output (None until a model provides it).
    Cl_err: float | None = None
    Cd_err: float | None = None
    Cm_err: float | None = None
    trusted: bool = True   # predicted error below the deployed gate → use directly
    ood: bool = False      # out-of-distribution → defer this evaluation to real CFD

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Surrogate(ABC):
    """Fast aerodynamic evaluator. Implementations: the trained Phase-2 model, or a stand-in.

    Contract: given airfoil coordinates (N,2, unit chord) and a flow condition, return an
    `AeroPrediction`. `predict_batch` defaults to a loop; a real model overrides it to
    vectorize. The confidence fields power the trust-gate and CFD fallback.
    """

    name: str = "surrogate"

    @abstractmethod
    def predict(self, coords: np.ndarray, alpha_deg: float, Re: float,
                mach: float = 0.0) -> AeroPrediction:
        ...

    def predict_batch(self, coords: np.ndarray, alphas, Re: float,
                      mach: float = 0.0) -> list[AeroPrediction]:
        return [self.predict(coords, float(a), Re, mach) for a in np.atleast_1d(alphas)]


class NeuralFoilSurrogate(Surrogate):
    """Placeholder `Surrogate` backed by NeuralFoil (a pretrained ML airfoil model).

    Lets the downstream be built/tested before our own surrogate + confidence model exist.
    NOT the final model and NOT trained on our data — a drop-in that satisfies the interface.
    NeuralFoil's own `analysis_confidence` is surfaced as a rough trust proxy until the real
    learned error model replaces it.
    """

    name = "neuralfoil-placeholder"

    def __init__(self, model_size: str = "medium", trust_threshold: float = 0.90):
        self.model_size = model_size
        self.trust_threshold = trust_threshold

    def predict(self, coords: np.ndarray, alpha_deg: float, Re: float,
                mach: float = 0.0) -> AeroPrediction:
        import neuralfoil as nf

        r = nf.get_aero_from_coordinates(
            coordinates=np.asarray(coords, dtype=float),
            alpha=alpha_deg, Re=Re, model_size=self.model_size,
        )
        conf = float(np.asarray(r.get("analysis_confidence", 1.0)).ravel()[0])
        return AeroPrediction(
            Cl=float(np.asarray(r["CL"]).ravel()[0]),
            Cd=float(np.asarray(r["CD"]).ravel()[0]),
            Cm=float(np.asarray(r["CM"]).ravel()[0]),
            trusted=conf >= self.trust_threshold,
            ood=conf < 0.5,
        )

    def predict_batch(self, coords: np.ndarray, alphas, Re: float,
                      mach: float = 0.0) -> list[AeroPrediction]:
        import neuralfoil as nf

        alphas = np.atleast_1d(alphas).astype(float)
        r = nf.get_aero_from_coordinates(
            coordinates=np.asarray(coords, dtype=float),
            alpha=alphas, Re=Re, model_size=self.model_size,
        )
        cl, cd, cm = np.asarray(r["CL"]), np.asarray(r["CD"]), np.asarray(r["CM"])
        conf = np.asarray(r.get("analysis_confidence", np.ones_like(cl)))
        return [AeroPrediction(Cl=float(cl[i]), Cd=float(cd[i]), Cm=float(cm[i]),
                               trusted=bool(conf[i] >= self.trust_threshold),
                               ood=bool(conf[i] < 0.5))
                for i in range(len(alphas))]


if __name__ == "__main__":  # smoke-check the interface + placeholder
    import aerosandbox as asb

    surr = NeuralFoilSurrogate()
    coords = asb.Airfoil("naca4412").coordinates
    for a in (-4, 0, 4, 8):
        p = surr.predict(coords, a, Re=1e6)
        print(f"a={a:+d}  Cl={p.Cl:+.3f}  Cd={p.Cd:.4f}  Cm={p.Cm:+.3f}  "
              f"trusted={p.trusted} ood={p.ood}")

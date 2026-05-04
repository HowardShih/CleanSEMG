"""
${CLEANSEMG_ROOT}/baseline_models/SDEMG.py

SDEMG — Score-Based Diffusion Model for sEMG denoising
(Liu et al., ICASSP 2024)

Wraps ConditionalModel + GaussianDiffusion1D from the SDEMG repo into the
standard baseline interface:
    model(noisy: Tensor[B, L]) -> pred: Tensor[B, L]   ← inference / forward
    model.compute_diffusion_loss(clean, noisy) -> loss  ← training

train_baseline.py detects HAS_DIFFUSION_LOSS = True and calls
compute_diffusion_loss() instead of F.l1_loss(model(noisy), clean).

SDEMG repo location (in priority order):
  1. Env var  SDEMG_REPO_PATH
  2. Sibling of the project root  (e.g. /…/SDEMG/ next to /…/SEMG/)
  3. Inside the project root      (e.g. /…/SEMG/SDEMG/)
"""

import os
import sys
import types
import torch
import torch.nn as nn

# ── locate SDEMG repo ──────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))   # …/SEMG/baseline_models/
_PROJECT_ROOT = os.path.dirname(_HERE)                        # …/SEMG/
_PARENT       = os.path.dirname(_PROJECT_ROOT)                # …/

_CANDIDATES = [
    os.environ.get("SDEMG_REPO_PATH", ""),          # 1. explicit env-var
    os.path.join(_PARENT, "SDEMG"),                 # 2. sibling of project root
    os.path.join(_PROJECT_ROOT, "SDEMG"),            # 3. inside project root
]

SDEMG_REPO: str = ""
for _c in _CANDIDATES:
    if _c and os.path.isdir(_c):
        SDEMG_REPO = _c
        break

if not SDEMG_REPO:
    raise RuntimeError(
        "[SDEMG] Cannot locate the SDEMG repo. Tried:\n"
        + "\n".join(f"  {c}" for c in _CANDIDATES if c)
        + "\nClone it or set:  export SDEMG_REPO_PATH=/path/to/SDEMG"
    )

if SDEMG_REPO not in sys.path:
    sys.path.insert(0, SDEMG_REPO)

# ── mock optional deps used only in __main__ blocks of SDEMG repo ─────────
# deep_filter_model.py does `from torchsummary import summary` at module
# level, but only uses it inside `if __name__ == '__main__':`.
# We inject a lightweight stub so the import succeeds without the package.
def _stub_module(name: str, **attrs) -> types.ModuleType:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]

_stub_module("torchsummary", summary=lambda *a, **kw: None)

# ── import SDEMG components ────────────────────────────────────────────────
try:
    from deep_filter_model import ConditionalModel   # noqa: E402
    from ddpm_1d import GaussianDiffusion1D          # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"[SDEMG] Import failed from repo '{SDEMG_REPO}': {exc}"
    ) from exc


# ── model ──────────────────────────────────────────────────────────────────
class SDEMG(nn.Module):
    """
    Baseline wrapper for SDEMG (score-based diffusion sEMG denoiser).

    Training
    --------
    Uses the diffusion MSE loss (pred_noise objective).  Detected by
    train_baseline.py via the HAS_DIFFUSION_LOSS class attribute.
    Call:  loss = model.compute_diffusion_loss(clean, noisy)

    Inference
    ---------
    Runs T reverse-diffusion steps conditioned on the noisy input.
    Call:  pred = model(noisy)         # standard forward()

    Shape contract (matches all other baseline models):
        noisy / clean : FloatTensor[B, L]  (no channel dim)
        pred          : FloatTensor[B, L]
    """

    HAS_DIFFUSION_LOSS: bool = True   # read by train_baseline._compute_loss()

    # ── defaults that match the paper / original default.yaml ────────────
    _SEQ_LEN   = 2000   # segment length used in this project (2 s @ 1 kHz)
    _TIMESTEPS = 50
    _FEATS     = 128

    def __init__(
        self,
        seq_length:        int  = _SEQ_LEN,
        timesteps:         int  = _TIMESTEPS,
        feats:             int  = _FEATS,
        objective:         str  = "pred_noise",
        loss_function:     str  = "l2",
        beta_schedule:     str  = "cosine",
        ddim:              bool = False,
        denoise_timesteps: int  = None,
    ):
        super().__init__()
        net = ConditionalModel(feats=feats)
        self.diffusion = GaussianDiffusion1D(
            net,
            seq_length    = seq_length,
            timesteps     = timesteps,
            objective     = objective,
            loss_function = loss_function,
            beta_schedule = beta_schedule,
            condition     = True,
        )
        self.ddim              = ddim
        self.denoise_timesteps = denoise_timesteps

    # ── training ──────────────────────────────────────────────────────────
    def compute_diffusion_loss(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion training / validation loss.

        Internally samples a random timestep t, adds noise to `clean`,
        and returns the MSE between the predicted and actual noise.

        Args:
            clean : FloatTensor[B, L]
            noisy : FloatTensor[B, L]  (ECG-contaminated, used as condition)
        Returns:
            scalar loss tensor
        """
        # GaussianDiffusion1D expects a channel dim → [B, 1, L]
        return self.diffusion(clean.unsqueeze(1), noisy.unsqueeze(1))

    # ── inference ─────────────────────────────────────────────────────────
    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Denoise via T reverse-diffusion steps.

        Args:
            noisy : FloatTensor[B, L]
        Returns:
            pred  : FloatTensor[B, L]
        """
        n = noisy.unsqueeze(1)      # [B, 1, L]
        if self.ddim:
            pred = self.diffusion.ddim_denoise(
                n, denoise_timesteps=self.denoise_timesteps
            )
        else:
            pred = self.diffusion.denoise(
                n, denoise_timesteps=self.denoise_timesteps
            )
        return pred.squeeze(1)      # [B, L]
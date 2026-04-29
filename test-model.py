"""Project smoke tests.

Goals:
1) Always run zero-dependency checks (compile + LexForge roundtrip).
2) Run import checks for every top-level module.
3) If PyTorch is installed, run lightweight model execution.
4) If PyTorch is missing, skip only torch-dependent steps (do not hide other errors).
"""

from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).name
PY_FILES = sorted(p for p in ROOT.glob("*.py") if p.name != SELF)


def compile_smoke() -> None:
    for path in PY_FILES:
        py_compile.compile(str(path), doraise=True)
    print(f"[ok] compiled {len(PY_FILES)} files")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def import_smoke(torch_available: bool) -> None:
    for mod_name in [p.stem for p in PY_FILES]:
        try:
            importlib.import_module(mod_name)
            print(f"[ok] imported {mod_name}")
        except ModuleNotFoundError as exc:
            if not torch_available and exc.name == "torch":
                print(f"[skip] {mod_name} requires torch")
                continue
            raise


def lexforge_roundtrip_smoke() -> None:
    from LexForge import LexForge

    tok = LexForge(vocab_size=2048)
    sample = "def f(x):\n    return x + 1\n"
    ids = tok.encode(sample)
    out = tok.decode(ids)
    assert out == sample, "LexForge roundtrip mismatch"
    print("[ok] LexForge roundtrip")


def torch_model_smoke() -> None:
    import torch
    from Architecture import CogForge, CogForgeConfig

    cfg = CogForgeConfig(vocab_size=256, max_seq_len=64, d_model=128, n_heads=4, n_kv_heads=2, n_layers=4)
    model = CogForge(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(ids, return_verifier=True)
    assert out["logits"].shape == (1, 16, cfg.vocab_size)
    print("[ok] torch forward")


def main() -> None:
    torch_available = has_torch()
    compile_smoke()
    import_smoke(torch_available=torch_available)
    lexforge_roundtrip_smoke()
    if torch_available:
        torch_model_smoke()
    else:
        print("[skip] torch model smoke (torch not installed)")
    print("[ok] smoke complete")


if __name__ == "__main__":
    main()

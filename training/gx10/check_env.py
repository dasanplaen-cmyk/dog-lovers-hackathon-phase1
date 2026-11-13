"""Environment check for the GX10. Prints versions, GPU and data-bundle presence; never data.

    .venv/bin/python training/gx10/check_env.py

Exit code 0 when everything needed for training is in place, 1 otherwise.
"""

import json
import os
import platform
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import games  # noqa: E402  (repo root on sys.path first)


def main():
    ok = True
    print("python   {} on {} ({})".format(platform.python_version(), platform.machine(),
                                          platform.platform()))
    for mod in ("numpy", "pandas", "kloppy", "yaml", "scipy"):
        try:
            m = __import__(mod)
            print("{:<8} {}".format(mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("{:<8} MISSING".format(mod))
            ok = False

    try:
        import torch
        print("torch    {} (CUDA build {})".format(torch.__version__, torch.version.cuda))
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info()
            x = torch.randn(1024, 1024, device="cuda")
            y = (x @ x.T).sum().item()  # forces a real kernel launch on this architecture
            print("gpu      {} sm_{}{} | {:.0f}/{:.0f} GiB free | matmul ok ({})".format(
                name, cap[0], cap[1], free / 2**30, total / 2**30, "finite" if y == y else "NaN"))
        elif os.environ.get("MTAH_CPU_ONLY"):
            print("gpu      not used (MTAH_CPU_ONLY): inference runs on CPU")
        else:
            print("gpu      NOT AVAILABLE to torch")
            ok = False
    except Exception as e:  # a wheel without kernels for this GPU fails here, not at import
        print("torch    FAILED: {}: {}".format(type(e).__name__, e))
        ok = False

    data_dir = os.path.realpath(games.DATA_DIR)
    inside = data_dir.startswith(os.path.realpath(REPO) + os.sep)
    print("data     MTAH_DATA_DIR={} ({})".format(
        data_dir, "inside repo, under gitignored data/" if inside else "outside repo"))
    present = {}
    for source in games.SOURCES:
        for name, path in games.asset_paths(source=source).items():
            present["{}/{}".format(source, name)] = os.path.isfile(path)
    print("bundle   " + json.dumps(present))
    # The videos only feed the optional video ball detector; training and inference on tracking
    # need the tactical tracking, homography and tagging.
    required = [k for k in present if k.startswith("tactical/") and not k.endswith("/video")]
    missing = [k for k in required if not present[k]]
    if missing:
        print("bundle   MISSING required: {}".format(", ".join(missing)))
        ok = False
    if not present.get("tactical/video"):
        print("bundle   note: no tactical video; only the video ball detector (P2) needs it")

    if not inside:
        settings = os.path.join(REPO, ".claude", "settings.local.json")
        covered = False
        if os.path.isfile(settings):
            with open(settings, encoding="utf-8") as f:
                deny = json.load(f).get("permissions", {}).get("deny", [])
            covered = "Read(/{}/**)".format(data_dir) in deny
        print("agents   deny rule for the bundle: {}".format("yes" if covered else "NO (run deny_data.py)"))
        ok = ok and covered

    print("RESULT   {}".format("OK" if ok else "NOT READY"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

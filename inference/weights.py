"""Published model weights: a manifest in git, the files in a GitHub Release.

The weights are trained on external games only, so they can be public: anyone with the data
bundle regenerates every file from them. inference/weights_manifest.json names each file, the run
it came from, its sha256 and the release tag.

    # once, where the weights are (GX10): copy them under release names and write the manifest
    python -m inference.weights manifest --tag weights-v1 \\
        ball_v1=data/work/weights/<ball run>/best.pt possession_v1=data/work/weights/<poss run>/best.pt
    gh release create weights-v1 data/work/weights/release/*.pt --title "Model weights v1" \\
        --notes "Trained on external games only. sha256 in inference/weights_manifest.json"

    # anywhere: download the release files and check their sha256
    python -m inference.weights fetch
    python -m inference.weights path possession_v1      # prints the verified local path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.runlog import work_dir  # noqa: E402

MANIFEST = os.path.join(REPO, "inference", "weights_manifest.json")
GITHUB_REPO = "dasanplaen-cmyk/dog-lovers-hackathon-phase1"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_dir() -> str:
    return os.path.join(str(work_dir()), "weights", "release")


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def local_path(name: str) -> str:
    """Verified local path of a published weights file (fetch it first)."""
    entry = load_manifest()["files"][name]
    path = os.path.join(release_dir(), entry["file"])
    if not os.path.isfile(path) or sha256(path) != entry["sha256"]:
        raise SystemExit("{} missing or wrong sha256; run: python -m inference.weights fetch".format(path))
    return path


def make_manifest(tag: str, pairs: list[str]) -> dict:
    import torch
    os.makedirs(release_dir(), exist_ok=True)
    files = {}
    if os.path.isfile(MANIFEST):        # keep earlier entries, each pinned to the release it is in
        old = load_manifest()
        for name, e in old["files"].items():
            files[name] = {**e, "tag": e.get("tag", old["tag"])}
    for pair in pairs:
        name, src = pair.split("=", 1)
        dst = os.path.join(release_dir(), name + ".pt")
        shutil.copyfile(src, dst)
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        files[name] = {"file": name + ".pt", "sha256": sha256(dst), "bytes": os.path.getsize(dst),
                       "run": ckpt.get("run"), "step": ckpt.get("step"), "tag": tag,
                       "trained_on": cfg.get("train_games"),
                       "selected_on": (cfg.get("val_games") or []) + (cfg.get("real_val_games") or [])}
    manifest = {"tag": tag, "repo": GITHUB_REPO, "files": files}
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def fetch() -> None:
    m = load_manifest()
    os.makedirs(release_dir(), exist_ok=True)
    for name, e in m["files"].items():
        path = os.path.join(release_dir(), e["file"])
        if os.path.isfile(path) and sha256(path) == e["sha256"]:
            print("ok    {}".format(e["file"]))
            continue
        tag = e.get("tag", m["tag"])
        got = False
        if shutil.which("gh"):   # works while the repository is private, if gh is logged in
            if os.path.exists(path):   # old gh versions have no --clobber: clear the stale file
                os.remove(path)
            got = subprocess.run(["gh", "release", "download", tag, "--repo", m["repo"],
                                  "--pattern", e["file"], "--dir", release_dir()]).returncode == 0
        if not got:              # public repository, or gh missing / not logged in
            url = "https://github.com/{}/releases/download/{}/{}".format(m["repo"], tag, e["file"])
            subprocess.run(["curl", "-sSfL", "-o", path, url], check=True)
        if sha256(path) != e["sha256"]:
            raise SystemExit("{}: sha256 mismatch after download".format(e["file"]))
        print("fetch {}".format(e["file"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--tag", required=True)
    m.add_argument("pairs", nargs="+", help="name=path/to/best.pt")
    sub.add_parser("fetch")
    p = sub.add_parser("path")
    p.add_argument("name")
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        print(json.dumps(make_manifest(args.tag, args.pairs), indent=2))
    elif args.cmd == "fetch":
        fetch()
    else:
        print(local_path(args.name))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
ChromaDB Orphan Cleanup
Removes Chroma state left behind by failed, aborted or deleted Caesar runs.

Run it and it surveys every store it can find, shows you exactly what it would
remove, and asks. Answer anything but "yes" and nothing happens.

Usage:
    python cleanup_chroma.py                       # scan the disk, then ask
    python cleanup_chroma.py --store rome          # just this one, no scan
    python cleanup_chroma.py --include run-dirs,unmatched
    python cleanup_chroma.py --path /some/chroma   # this dir, no discovery
    python cleanup_chroma.py --json                # report only, no prompt
    python cleanup_chroma.py --force               # unattended; answers yes

Stores are located from configuration rather than guesswork -- instantly, and
correctly even on a relocated checkout:

  rome  ~/.rome/agent-chroma-db   per rome/kb_server.py::CHROMA_BASE_DIR
  web   <data>/chroma             per the web server's CAESAR_WEB_DATA_DIR,
                                  read from the environment or web_server/.env

A store counts as a web-server store when caesar_web.sqlite sits beside it, so
run-aware classification follows the data instead of a hardcoded path. Running
with no arguments also sweeps the filesystem for stores those rules miss; that
costs about a minute, so naming --store or --path skips it. Works on macOS and
Linux, and the only external commands are find, lsof and pgrep, each optional.

If a store is open -- the web server holds its own -- the steps that write to
sqlite are deferred and the rest still run, since moving directories Chroma has
no handle on is safe underneath a live server.

Stores accumulate three kinds of debris:

  1. HNSW index directories whose `segments` row is gone. Chroma's
     delete_collection() drops the SQLite rows but leaves the persist dir.
  2. Collections holding zero embeddings -- created, then the run died before
     writing anything.
  3. Collections holding data whose run artifacts no longer exist on disk.

Every individual item is printed before the prompt. On a yes, the store is
backed up, the plan is rehearsed against a throwaway copy, and everything
removed is moved into the backup rather than unlinked, so a run is reversible.

Deleting a collection can orphan its `mem0_<name>_agent_*` shadow, which then
surfaces as `unmatched` on the *next* run rather than cascading automatically.
That is deliberate: a shadow may hold real memories and deserves a second look.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
# This file lives at <repo>/monitor/, so the repo root is one level up. Deriving
# it beats hardcoding a checkout location and works wherever rome is cloned.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Roots searched for run artifacts. Point at a sibling or home-directory checkout
# of whichever pipeline produced them via --artifact-root, or set ROME_ARTIFACT_ROOTS
# to a colon-separated list.
DEFAULT_ARTIFACT_ROOTS = [
    p for p in (Path(r).expanduser()
                for r in os.environ.get("ROME_ARTIFACT_ROOTS", "").split(os.pathsep) if r)
    if p.exists()
]

# Directory names never worth descending into during --scan. Kept conservative:
# a Chroma store inside any of these is not something to clean automatically.
SCAN_PRUNE = [
    "node_modules", "site-packages", ".venv", "venv", ".git", ".Trash", ".cache",
    "Library", "Applications", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".npm", ".cargo", ".rustup", ".conda", "miniconda3", "anaconda3", "build",
    "dist", ".next", ".gradle", ".m2", "target",
]

UUID_RE = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
# Per-question collections written by the task_hle / task_fp pipelines. Only
# names matching this are eligible for automatic dead-run deletion -- we know
# exactly which directory should exist for them.
QUESTION_RE = re.compile(r"^q\d+_[0-9a-f]{6,}(_[0-9a-f]{6,})?$")
# Plain `caesar` CLI runs. run_agent.py::resolve_experiment_repository names the
# run directory <date>_<config>_q-<hash>_t-<iters>[_id-<n>] and the collection
# takes the same name, so the directory is an exact lookup.
CLI_RUN_RE = re.compile(r"^\d{2}-\d{2}-\d{2}_.+_q-[0-9a-f]{6,}_t-\d+(_id-.+)?$")
MEM0_RE = re.compile(r"^mem0_(?P<base>.+)_agent_[A-Za-z0-9_]+$")
WEB_RUN_RE = re.compile(r"^(?:web|mem0)_(?P<run>[0-9a-f]{32})(?:_agent_[A-Za-z0-9_]+)?$")
PRUNE_DIRS = {".git", "node_modules", ".venv", "__pycache__", "paper", ".mypy_cache"}

# Finding kinds, in the order they are reported.
ORPHAN_DIR = "orphan-index-dir"
EMPTY_ORPHAN = "empty-collection-orphan"
EMPTY_LIVE = "empty-collection-live"
DEAD_RUN = "dead-run-collection"
UNMATCHED = "unmatched-collection"
ORPHAN_RUN_DIR = "orphan-run-dir"

# Removed by default; the rest need naming in --include.
DEFAULT_KINDS = {ORPHAN_DIR, EMPTY_ORPHAN, DEAD_RUN}
OPT_IN = {
    "empty-live": EMPTY_LIVE,
    "unmatched": UNMATCHED,
    "run-dirs": ORPHAN_RUN_DIR,
}


@dataclass
class Finding:
    kind: str
    name: str          # collection name, or directory basename
    reason: str
    bytes: int = 0
    collection_id: str | None = None
    paths: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "reason": self.reason,
            "bytes": self.bytes,
            "collection_id": self.collection_id,
            "paths": [str(p) for p in self.paths],
        }


@dataclass
class Survey:
    store: str
    path: Path
    findings: list[Finding]
    n_collections: int
    n_queue: int
    purgeable_now: int

    @property
    def db(self) -> Path:
        return self.path / "chroma.sqlite3"


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "K", "M"):
        if n < 1024:
            return f"{n:.0f}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


def caesar_result_roots() -> list[Path]:
    """Where plain `caesar` CLI runs write their output.

    Mirrors caesar/run_agent.py::default_result_root. That function picks one
    root; we consider every candidate that exists, because a run left behind by
    an older layout must not be mistaken for a dead one.
    """
    env = os.environ.get("CAESAR_RESULT_DIR")
    if env:
        return [Path(env).expanduser().resolve()]
    return [p for p in (REPO_ROOT / "caesar" / "result",
                        Path("~/.caesar/result").expanduser()) if p.is_dir()]


@dataclass
class Artifacts:
    """Run directory names, kept apart by the layout that produced them.

    Keeping the two families separate is a safety property, not tidiness. A
    collection is only judged against the family it belongs to, and only when
    that family turned up something -- so an index that found CLI runs but no
    pipeline runs cannot conclude that every pipeline collection is dead.
    """
    pipeline: set[str] = field(default_factory=set)   # dirs under caesar_runs/
    cli: set[str] = field(default_factory=set)        # dirs under a result root

    def __bool__(self) -> bool:
        return bool(self.pipeline or self.cli)


def index_artifacts(roots: list[Path], cli_roots: list[Path]) -> Artifacts:
    """Directory names a collection can be matched against.

    The task_hle / task_fp pipelines nest a directory per question under
    `caesar_runs/`; the CLI writes one directory per run directly under its
    result root. Both name the directory exactly as the collection, so
    membership means the run still exists.
    """
    art = Artifacts()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
            if Path(dirpath).name == "caesar_runs":
                art.pipeline.update(dirnames)
    for root in cli_roots:
        art.cli.update(c.name for c in root.iterdir() if c.is_dir())
    return art


def read_collections(db: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT c.id,
                   c.name,
                   (SELECT s.id FROM segments s
                     WHERE s.collection = c.id AND s.scope = 'VECTOR') AS vseg,
                   (SELECT COUNT(*) FROM embeddings e
                     WHERE e.segment_id = (SELECT s.id FROM segments s
                                            WHERE s.collection = c.id
                                              AND s.scope = 'METADATA')) AS n_emb
              FROM collections c
             ORDER BY c.name
            """
        ).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "name": r[1], "vseg": r[2], "n_emb": r[3]} for r in rows]


def read_scalar(db: Path, sql: str) -> int:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchone()[0] or 0
    finally:
        con.close()


def web_app_db(store_path: Path) -> Path:
    """The web server keeps caesar_web.sqlite beside its chroma/ persist dir."""
    return store_path.parent / "caesar_web.sqlite"


def web_run_dir(store_path: Path) -> Path:
    return store_path.parent / "runs"


def web_live_runs(app_db: Path) -> tuple[set[str], bool]:
    """Run ids known to caesar_web.sqlite. Second value is False if unreadable."""
    if not app_db.exists():
        return set(), False
    con = sqlite3.connect(f"file:{app_db}?mode=ro", uri=True)
    try:
        return {r[0].replace("-", "") for r in con.execute("SELECT id FROM runs")}, True
    except sqlite3.Error:
        return set(), False
    finally:
        con.close()


def known_stores() -> dict[str, Path]:
    """Stores locatable from configuration alone, with no filesystem walk.

    `rome` follows rome/kb_server.py::CHROMA_BASE_DIR. `web` follows the web
    server's CAESAR_WEB_DATA_DIR setting, whose default is <repo>/web_server/
    api/data -- read from the environment or the repo's .env files so a
    relocated data dir is picked up instead of guessed at.
    """
    found: dict[str, Path] = {}

    agent = Path("~/.rome/agent-chroma-db").expanduser()
    if (agent / "chroma.sqlite3").exists():
        found["rome"] = agent

    data_dir = os.environ.get("CAESAR_WEB_DATA_DIR")
    if not data_dir:
        for env_file in (REPO_ROOT / "web_server" / ".env.local",
                         REPO_ROOT / "web_server" / ".env"):
            if not env_file.exists():
                continue
            for line in env_file.read_text(errors="replace").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "CAESAR_WEB_DATA_DIR" and value.strip():
                    data_dir = value.strip().strip("\'\"")
                    break
            if data_dir:
                break
    web = (Path(data_dir).expanduser() if data_dir
           else REPO_ROOT / "web_server" / "api" / "data") / "chroma"
    if (web / "chroma.sqlite3").exists():
        found["web"] = web

    return found


def scan_stores(roots: list[Path]) -> dict[str, Path]:
    """Walk for chroma.sqlite3 using find(1), which POSIX guarantees on both
    macOS and Linux and which is far quicker than os.walk over a large tree.

    Our own timestamped backups are skipped -- they are byte-identical copies
    of a store and cleaning one would be pointless at best.
    """
    found: dict[str, Path] = {}
    prune = []
    for name in SCAN_PRUNE + ["*-backup-*"]:
        prune += ["-name", name, "-o"]
    for root in roots:
        if not root.exists():
            continue
        # -xdev keeps the walk on one filesystem: network shares and external
        # volumes are not somewhere to go hunting for stores to delete.
        cmd = ["find", str(root), "-xdev", "("] + prune[:-1] + [")", "-prune",
               "-o", "-name", "chroma.sqlite3", "-print"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"WARNING: scan of {root} failed: {exc}", file=sys.stderr)
            continue
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            path = Path(line.strip()).parent.resolve()
            if path not in found.values():
                found[label_for(path, found)] = path
    return found


def label_for(path: Path, taken: dict[str, Path]) -> str:
    """A short, stable, unique display name for a discovered store."""
    for label, known in known_stores().items():
        if known.resolve() == path:
            return label
    base = path.name if path.name != "chroma" else path.parent.name
    label, n = base, 2
    while label in taken and taken[label] != path:
        label, n = f"{base}-{n}", n + 1
    return label


def survey(store: str, path: Path, artifacts: Artifacts) -> Survey:
    db = path / "chroma.sqlite3"
    if not db.exists():
        raise SystemExit(f"no chroma.sqlite3 under {path}")

    collections = read_collections(db)
    by_name = {c["name"]: c for c in collections}
    live_segments = set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        live_segments = {r[0] for r in con.execute("SELECT id FROM segments")}
    finally:
        con.close()

    findings: list[Finding] = []

    # 1. index directories with no owning segment row
    for child in sorted(path.iterdir()):
        if child.is_dir() and UUID_RE.match(child.name) and child.name not in live_segments:
            findings.append(
                Finding(
                    ORPHAN_DIR,
                    child.name,
                    "HNSW index dir with no `segments` row (collection was deleted)",
                    dir_size(child),
                    paths=[child],
                )
            )

    # A web-server store always sits at <data>/chroma with caesar_web.sqlite as
    # its sibling. Detecting that is what enables run-aware classification --
    # far sturdier than matching on a label or a hardcoded path.
    app_db, run_dir = web_app_db(path), web_run_dir(path)
    is_web = app_db.exists()
    run_ids, app_db_ok = web_live_runs(app_db) if is_web else (set(), True)

    def artifacts_exist(name: str) -> bool | None:
        """True/False if we can decide, None if the name tells us nothing.

        Checked against the full collection name first: on the web server both
        `web_<runid>` and `mem0_<runid>_agent_X` carry the run id directly.
        """
        if is_web:
            m = WEB_RUN_RE.match(name)
            if m:
                run = m.group("run")
                return run in run_ids or (run_dir / run).is_dir()
            return None
        # Judge a name only against its own family, and only if that family
        # produced any artifacts at all. Otherwise "the run is gone" and "we
        # never looked where these live" are indistinguishable, and guessing
        # wrong here deletes live research data.
        if QUESTION_RE.match(name):
            return name in artifacts.pipeline if artifacts.pipeline else None
        if CLI_RUN_RE.match(name):
            return name in artifacts.cli if artifacts.cli else None
        return None

    for col in collections:
        name, n_emb = col["name"], col["n_emb"]
        alive = artifacts_exist(name)
        if alive is None:
            # A memory collection lives and dies with the collection it shadows.
            m = MEM0_RE.match(name)
            if m:
                base = m.group("base")
                alive = artifacts_exist(base)
                if alive is None and base in by_name:
                    alive = True  # shadowed collection is still present in this store

        if n_emb == 0:
            if alive:
                findings.append(
                    Finding(EMPTY_LIVE, name,
                            "0 embeddings, but its run still exists",
                            collection_id=col["id"]))
            else:
                findings.append(
                    Finding(EMPTY_ORPHAN, name,
                            "0 embeddings and no surviving run artifacts",
                            collection_id=col["id"]))
            continue

        if alive is False:
            findings.append(
                Finding(DEAD_RUN, name,
                        f"{n_emb} embeddings but its run directory is gone",
                        dir_size(path / col["vseg"]) if col["vseg"] and (path / col["vseg"]).is_dir() else 0,
                        collection_id=col["id"],
                        paths=[path / col["vseg"]] if col["vseg"] else []))
        elif alive is None:
            findings.append(
                Finding(UNMATCHED, name,
                        f"{n_emb} embeddings; name does not map to a known run layout",
                        collection_id=col["id"]))

    # 2. web run directories with no row in the app DB
    if is_web and app_db_ok and run_dir.is_dir():
        for child in sorted(run_dir.iterdir()):
            if child.is_dir() and child.name not in run_ids:
                findings.append(
                    Finding(ORPHAN_RUN_DIR, child.name,
                            "run output directory with no row in caesar_web.sqlite",
                            dir_size(child), paths=[child]))

    order = [ORPHAN_DIR, EMPTY_ORPHAN, DEAD_RUN, EMPTY_LIVE, UNMATCHED, ORPHAN_RUN_DIR]
    findings.sort(key=lambda f: (order.index(f.kind), f.name))

    return Survey(
        store=store,
        path=path,
        findings=findings,
        n_collections=len(collections),
        n_queue=read_scalar(db, "SELECT COUNT(*) FROM embeddings_queue"),
        purgeable_now=read_scalar(
            db,
            "SELECT COUNT(*) FROM embeddings_queue "
            "WHERE seq_id < (SELECT COALESCE(MIN(seq_id), 0) FROM max_seq_id)",
        ),
    )


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------

def holders(db: Path, is_web: bool) -> list[str]:
    """Processes with the store open, plus the web server if it is up.

    Chroma opens SQLite lazily, so lsof alone can come back empty while the
    server is very much alive and one request away from writing.
    """
    found = []
    try:
        out = subprocess.run(["lsof", "-t", str(db)], capture_output=True, text=True, timeout=15)
        found += [p for p in out.stdout.split() if p.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    if is_web:
        try:
            out = subprocess.run(["pgrep", "-f", "uvicorn app.main:app"],
                                 capture_output=True, text=True, timeout=15)
            found += [p for p in out.stdout.split() if p.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(set(found))


def backup(sv: Survey) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = sv.path.parent / f"{sv.path.name}-backup-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(sv.db) + suffix)
        if src.exists():
            shutil.copy2(src, dest / src.name)
    return dest


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def delete_collections(persist: Path, names: list[str]) -> tuple[int, list[str]]:
    """Delete via the Chroma API so metadata, FTS and WAL rows stay consistent."""
    if not names:
        return 0, []
    import chromadb

    client = chromadb.PersistentClient(path=str(persist))
    done, failed = 0, []
    for name in names:
        try:
            client.delete_collection(name)
            done += 1
        except Exception as exc:  # a missing collection is not worth aborting for
            failed.append(f"{name}: {exc}")
    return done, failed


def purge_and_vacuum(persist: Path) -> str:
    """Drop WAL rows every segment has consumed, then reclaim the freed pages.

    Chroma 1.5's Rust bindings expose no purge/vacuum entry point, so this
    applies Chroma's own criterion directly. Note the floor is the *minimum*
    max_seq_id across all segments: a write-once collection that is never
    touched again pins the WAL for the entire store, so the yield here can be
    small even when the queue is large. That is inherent, not corruption.
    """
    con = sqlite3.connect(persist / "chroma.sqlite3", isolation_level=None)
    try:
        floor = con.execute("SELECT COALESCE(MIN(seq_id), 0) FROM max_seq_id").fetchone()[0]
        con.execute("BEGIN")
        removed = con.execute(
            "DELETE FROM embeddings_queue WHERE seq_id < ?", (floor,)
        ).rowcount
        con.execute("COMMIT")
        con.execute("VACUUM")
        return f"purged {removed} rows below seq {floor}, vacuumed"
    finally:
        con.close()


def sweep_orphan_dirs(persist: Path, dest: Path | None) -> list[str]:
    """Move index dirs with no `segments` row into dest (or delete if dest is None)."""
    con = sqlite3.connect(f"file:{persist / 'chroma.sqlite3'}?mode=ro", uri=True)
    try:
        live = {r[0] for r in con.execute("SELECT id FROM segments")}
    finally:
        con.close()
    moved = []
    for child in sorted(persist.iterdir()):
        if child.is_dir() and UUID_RE.match(child.name) and child.name not in live:
            if dest is None:
                shutil.rmtree(child)
            else:
                shutil.move(str(child), str(dest / child.name))
            moved.append(child.name)
    return moved


def rehearse(sv: Survey, names: list[str], do_wal: bool) -> None:
    """Run the whole plan against a throwaway copy before touching the real store."""
    with tempfile.TemporaryDirectory(prefix="chroma-rehearsal-") as tmp:
        copy = Path(tmp) / sv.path.name
        shutil.copytree(sv.path, copy, ignore=shutil.ignore_patterns("*.log", "*-backup-*"))
        _, failed = delete_collections(copy, names)
        if failed:
            raise SystemExit("rehearsal failed, real store untouched:\n  " + "\n  ".join(failed))
        if do_wal:
            purge_and_vacuum(copy)
        con = sqlite3.connect(copy / "chroma.sqlite3")
        try:
            result = con.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            con.close()
        if result != "ok":
            raise SystemExit(f"rehearsal integrity_check returned {result!r}; real store untouched")


def apply(sv: Survey, selected: list[Finding], do_wal: bool) -> None:
    names = [f.name for f in selected if f.collection_id]
    moves = [f for f in selected if not f.collection_id and f.paths]

    # Rehearsing is only worth the copy when rows are being deleted; moving
    # untracked directories has nothing to get wrong.
    if names:
        print(f"  rehearsing {len(names)} collection deletions on a temporary copy ...")
        rehearse(sv, names, do_wal)
        print("  rehearsal clean")

    dest = backup(sv)
    print(f"  backup: {dest}")

    for f in moves:
        for p in f.paths:
            if p.exists():
                shutil.move(str(p), str(dest / p.name))
    if moves:
        print(f"  moved {len(moves)} director{'y' if len(moves) == 1 else 'ies'} into the backup")

    done, failed = delete_collections(sv.path, names)
    print(f"  deleted {done} collection(s)")
    for err in failed:
        print(f"  ! {err}")

    # delete_collection() drops the rows but leaves the HNSW dir on disk -- the
    # very leak this tool exists to clean. Sweep what we just orphaned so a
    # single pass finishes the job instead of needing a second run.
    if done:
        swept = sweep_orphan_dirs(sv.path, dest)
        if swept:
            print(f"  swept {len(swept)} index dir(s) orphaned by those deletions")

    if do_wal:
        before = read_scalar(sv.db, "SELECT COUNT(*) FROM embeddings_queue")
        size_before = sv.db.stat().st_size
        note = purge_and_vacuum(sv.path)
        after = read_scalar(sv.db, "SELECT COUNT(*) FROM embeddings_queue")
        size_after = sv.db.stat().st_size
        print(f"  WAL {before} -> {after} rows, sqlite {human(size_before)} -> "
              f"{human(size_after)} ({note})")

    (dest / "manifest.json").write_text(
        json.dumps({"store": sv.store, "removed": [f.as_dict() for f in selected]}, indent=2)
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(sv: Survey) -> None:
    print(f"\n=== {sv.store}  {sv.path}")
    print(f"    {sv.n_collections} collections, {sv.n_queue} WAL rows "
          f"({sv.purgeable_now} purgeable before cleanup), sqlite {human(sv.db.stat().st_size)}")
    if not sv.findings:
        print("    nothing orphaned")
        return

    for kind in (ORPHAN_DIR, EMPTY_ORPHAN, DEAD_RUN, EMPTY_LIVE, UNMATCHED, ORPHAN_RUN_DIR):
        group = [f for f in sv.findings if f.kind == kind]
        if not group:
            continue
        total = sum(f.bytes for f in group)
        mark = "will ask" if kind in OPT_IN.values() else "remove"
        print(f"\n  [{mark}] {kind} -- {len(group)} item(s)"
              + (f", {human(total)}" if total else ""))
        shown = group if len(group) <= 8 else group[:6]
        for f in shown:
            size = f", {human(f.bytes)}" if f.bytes else ""
            print(f"      {f.name}{size}")
            print(f"        {f.reason}")
        if len(shown) < len(group):
            print(f"      ... and {len(group) - len(shown)} more (--verbose to list)")


def manifest(sv: Survey, selected: list[Finding], do_wal: bool) -> int:
    """Print every individual item that --apply would remove. No truncation."""
    print(f"\n{'=' * 72}\nPLAN for {sv.store}  ({sv.path})\n{'=' * 72}")

    if not selected:
        print("  no collections or directories to remove")
    for kind in (ORPHAN_DIR, EMPTY_ORPHAN, DEAD_RUN, EMPTY_LIVE, UNMATCHED, ORPHAN_RUN_DIR):
        group = [f for f in selected if f.kind == kind]
        if not group:
            continue
        print(f"\n  {kind}  ({len(group)} item(s), {human(sum(f.bytes for f in group))})")
        for f in group:
            verb = "DELETE collection" if f.collection_id else "MOVE to backup  "
            print(f"    {verb}  {f.name}" + (f"  [{human(f.bytes)}]" if f.bytes else ""))
            for p in f.paths:
                print(f"                      {p}")

    reclaim = sum(f.bytes for f in selected)
    if do_wal:
        floor = read_scalar(sv.db, "SELECT COALESCE(MIN(seq_id), 0) FROM max_seq_id")
        rows = read_scalar(sv.db, f"SELECT COUNT(*) FROM embeddings_queue WHERE seq_id < {floor}")
        print(f"\n  write-ahead log\n    PURGE  {rows} of {sv.n_queue} WAL rows "
              f"(consumed by every segment, i.e. below seq {floor}), then VACUUM")
        if rows < sv.n_queue * 0.5:
            print("    note: the floor is the lowest max_seq_id across ALL segments, so a "
                  "write-once\n          collection that is never updated pins the rest. "
                  "That is normal, not corruption.")

    chosen = {f.kind for f in selected}
    skipped = [f"{k} ({len([f for f in sv.findings if f.kind == k])})"
               for k in OPT_IN.values()
               if k not in chosen and any(f.kind == k for f in sv.findings)]
    print(f"\n  NOT touched: {'; '.join(skipped) if skipped else 'nothing else flagged'}")
    print(f"  Reversible:  collections + directories are copied into a timestamped "
          f"backup\n               beside {sv.path.name} before anything is removed.")
    print(f"\n  Disk reclaimed on removal: {human(reclaim)} "
          f"(plus whatever VACUUM frees inside the {human(sv.db.stat().st_size)} sqlite)")
    return reclaim


def ask_group(store: str, token: str, group: list[Finding]) -> bool:
    """Offer one opt-in group, itemised, at the moment of the decision."""
    total = sum(f.bytes for f in group)
    print(f"\n  {store}: {len(group)} {token} item(s)"
          + (f", {human(total)}" if total else "") + " --")
    for f in group:
        print(f"      {f.name}" + (f"  [{human(f.bytes)}]" if f.bytes else ""))
        print(f"        {f.reason}")
    try:
        reply = input("  Remove these too? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reply.strip().lower() in ("y", "yes")


def confirm(store: str) -> bool:
    try:
        reply = input(f"\nProceed with cleanup of '{store}'? Type 'yes' to continue: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reply.strip().lower() == "yes"


def resolve_stores(args) -> dict[str, Path]:
    """Work out which stores to consider: explicit paths, a scan, or config.

    Config-derived lookup is the default because it is instant and exact. A
    scan is opt-in since walking a home directory costs seconds and can turn up
    stores that belong to something else entirely.
    """
    if args.path:
        out = {}
        for d in args.path:
            d = d.expanduser().resolve()
            if not (d / "chroma.sqlite3").exists():
                raise SystemExit(f"no chroma.sqlite3 under {d}")
            out[label_for(d, out)] = d
        return out

    stores = known_stores()
    if not args.no_scan:
        print(f"scanning {HOME} for Chroma stores; this takes a minute. "
              "--no-scan skips it.")
        for label, path in scan_stores([HOME]).items():
            if path not in stores.values():
                stores[label] = path
    return stores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", action="append", type=Path, default=None, metavar="DIR",
                    help="use these Chroma persist dirs verbatim (repeatable); "
                         "skips discovery entirely")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the filesystem sweep and use only the stores that "
                         "configuration points at (instant)")
    ap.add_argument("--artifact-root", action="append", type=Path, default=None,
                    metavar="DIR", help="where run directories live, if not in the "
                                        "usual places (repeatable)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="unattended: remove the safe groups without prompting, "
                         "and leave the opt-in groups alone")
    args = ap.parse_args()

    roots = args.artifact_root or DEFAULT_ARTIFACT_ROOTS
    for root in roots:
        if not root.exists():
            print(f"WARNING: artifact root {root} does not exist", file=sys.stderr)
    cli_roots = caesar_result_roots()
    artifacts = index_artifacts(roots, cli_roots)
    where = [str(r) for r in roots] + [f"{r} (CLI runs)" for r in cli_roots]
    print(f"indexed {len(artifacts.pipeline)} pipeline and {len(artifacts.cli)} CLI run "
          f"directories under " + (", ".join(where) if where else "nowhere"))
    for label, names, hint in (("pipeline", artifacts.pipeline, "--artifact-root DIR"),
                               ("CLI", artifacts.cli, "CAESAR_RESULT_DIR")):
        if not names:
            print(f"NOTE: no {label} run directories found, so {label} collections are "
                  f"reported as\n      unmatched rather than judged dead. Point at them "
                  f"with {hint} if\n      that is wrong.")

    stores = resolve_stores(args)
    if not stores:
        print("no Chroma stores found. Name one with --path DIR", file=sys.stderr)
        return 1
    surveys = [survey(name, path, artifacts) for name, path in stores.items()]

    for sv in surveys:
        report(sv)

    interactive = sys.stdin.isatty()
    if not interactive and not args.force:
        print("\nreport only -- stdin is not a terminal, so nothing was prompted for "
              "or removed.\nRe-run in a terminal, or pass --force to act unattended.")
        return 0

    for sv in surveys:
        # Groups that are always safe to remove, plus any opt-in group the user
        # accepts. Asking beats a flag here: the items are on screen, named, at
        # the moment of the decision.
        kinds = set(DEFAULT_KINDS)
        for token, kind in OPT_IN.items():
            group = [f for f in sv.findings if f.kind == kind]
            if not group:
                continue
            if args.force:
                # --force is for unattended runs, where the safe reading of an
                # opt-in group is "leave it". These need eyes, not automation.
                print(f"  {sv.store}: leaving {len(group)} {token} item(s) alone "
                      f"(--force never takes opt-in groups)")
                continue
            if ask_group(sv.store, token, group):
                kinds.add(kind)

        selected = [f for f in sv.findings if f.kind in kinds]
        do_wal = sv.purgeable_now > 0
        if not selected and not do_wal:
            continue

        # Only steps that write to sqlite care whether the store is open. When
        # it is, drop those and keep the rest rather than refusing outright:
        # moving directories Chroma has no handle on is safe under a live server.
        n_deletes = sum(1 for f in selected if f.collection_id)
        writes = ([f"{n_deletes} collection deletion(s)"] if n_deletes else []) \
            + (["WAL purge + VACUUM"] if do_wal else [])
        busy = holders(sv.db, web_app_db(sv.path).exists()) if writes else []
        if busy:
            print(f"\n  {sv.store} is held open by pid(s) {', '.join(busy)}, so "
                  f"{' and '.join(writes)}\n  must wait. Stop the holder and re-run for those. "
                  f"Continuing with the\n  parts that write no sqlite.")
            do_wal = False
            selected = [f for f in selected if not f.collection_id]
            if not selected:
                print(f"  nothing left to do on {sv.store} while it is open.")
                continue

        manifest(sv, selected, do_wal)
        if not args.force and not confirm(sv.store):
            print(f"  skipped, {sv.store} untouched")
            continue

        print(f"\n--- cleaning {sv.store}")
        apply(sv, selected, do_wal)

    return 0


if __name__ == "__main__":
    sys.exit(main())

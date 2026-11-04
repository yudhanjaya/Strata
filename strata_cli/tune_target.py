"""Turn what a person types after `strata tune` into the saved setup to tune: `strata tune iq3_s` instead of a profile name or a
path to a JSON file.

The things a name can mean, in the order they are tried:

1. an exact saved profile (`launch-iq3_s-128k`), which is what the command always accepted;
2. an installed model (`iq3_s`, from `strata models list`). If saved setups of that model exist (what `strata launch` writes: a context,
   cards, picture reading), the one used last is tuned, because a tune belongs to the choices it was made for, and the others are named;
3. a part of a name (`xxs`, `swift`), when exactly one saved setup or model contains it.

Anything else gets the list of what exists and the closest names, never a guess.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class TargetError(ValueError):
    """The name does not point at exactly one thing. The message is for the user."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = code


@dataclass
class Choice:
    ref: str                         # what `tune auto` is given: a profile id, or the path of a run configuration
    id: str
    kind: str                        # profile | model
    note: str = ""                   # one line for the person when more than one thing matched
    others: list[str] = field(default_factory=list)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _bare(profile_id: str) -> str:
    low = profile_id.lower()
    return low[len("launch-"):] if low.startswith("launch-") else low


def _of_model(profile_id: str, model_id: str) -> bool:
    """Is this saved setup one of that model's? `launch-iq3_s-128k` belongs to `iq3_s`, `launch-iq3_xxs-128k` does not."""
    rest = _bare(profile_id)
    m = model_id.lower()
    return rest.startswith(m) and (len(rest) == len(m) or rest[len(m)] in "-_ .")


def choose(query: str, profiles: list[str], models: dict[str, str], *, newest: Callable[[str], float] = lambda p: 0.0) -> Choice:
    """Pick the one thing ``query`` means. ``profiles`` are saved profile ids, ``models`` maps an installed model id to its run
    configuration's path, ``newest`` gives a profile's last-used time (larger is more recent)."""
    q = query.strip()
    if not q:
        raise TargetError("say which model or saved setup to tune, for example: strata tune iq3_s")
    low = q.lower()

    # 1. an exact saved profile (also with the `launch-` that `strata launch` puts in front)
    for pid in profiles:
        if pid.lower() in (low, "launch-" + low):
            return Choice(pid, pid, "profile")

    # 2. an installed model: its newest saved setup, or the model's own run configuration
    exact_models = [m for m in models if m.lower() == low or norm(m) == norm(q)]
    if exact_models:
        model = exact_models[0]
        setups = sorted((p for p in profiles if _of_model(p, model)), key=newest, reverse=True)
        if setups:
            others = setups[1:]
            note = f"Tuning your saved setup '{setups[0]}' (the one used last for {model})."
            if others:
                note += " Others: " + ", ".join(others) + ". Tune one of those with: strata tune <name>"
            return Choice(setups[0], setups[0], "profile", note, others)
        return Choice(models[model], model, "model", f"Tuning {model} with its installed settings (no saved setup of it was found).")

    # 3. a part of one name
    key = norm(q)
    part_p = [p for p in profiles if key and key in norm(p)]
    part_m = [m for m in models if key and key in norm(m)]
    if len(part_m) == 1 and all(_of_model(p, part_m[0]) for p in part_p):
        return choose(part_m[0], profiles, models, newest=newest)       # one model, and only its own saved setups besides
    if not part_m and len(part_p) == 1:
        return Choice(part_p[0], part_p[0], "profile")

    everything = sorted(set(profiles) | set(models))
    if part_p or part_m:
        raise TargetError(f"'{query}' matches more than one: " + ", ".join(sorted(part_p + part_m))
                          + ". Use the full name, for example: strata tune " + (part_p + part_m)[0])
    close = difflib.get_close_matches(low, [e.lower() for e in everything], n=3, cutoff=0.5)
    close = [e for e in everything if e.lower() in close]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    have = ("Saved setups and installed models: " + ", ".join(everything)) if everything else \
        "Nothing is installed yet. Run 'strata launch' to choose and set up a model."
    raise TargetError(f"no saved setup or installed model is called '{query}'.{hint} {have}")


def resolve(query: str) -> Choice:
    """`choose` over what is on this computer."""
    from strata_control import models as models_mod, profiles

    saved: list[str] = []
    stamp: dict[str, float] = {}
    for item in profiles.list_profiles():
        if item.get("error"):
            continue
        pid = item["id"]
        saved.append(pid)
        for p in (item.get("source_config"), str(profiles.profile_dir() / f"{pid}.json")):
            try:
                stamp[pid] = max(stamp.get(pid, 0.0), Path(p).stat().st_mtime)      # a launch rewrites its run configuration
            except (OSError, TypeError):
                pass
    found: dict[str, str] = {}
    try:
        for m in models_mod.list_installed().get("models") or []:
            if m.get("id") and m.get("source_config") and m.get("component_status") != "invalid":
                found.setdefault(m["id"], m["source_config"])
    except Exception:  # noqa: BLE001 - a model folder that cannot be read must not hide the saved setups
        pass
    return choose(query, saved, found, newest=lambda p: stamp.get(p, 0.0))


#: The words `strata tune` already has; anything else after it is a model or setup to tune.
SUBCOMMANDS = frozenset({"plan", "run", "apply", "rollback", "status", "list", "auto", "report"})


def rewrite_argv(argv: list[str]) -> list[str]:
    """`strata tune iq3_s --mode thorough` is `strata tune auto iq3_s --mode thorough`. Only a plain word that is not one of the
    command's own subcommands is taken for a name, so every existing command line keeps its meaning."""
    if len(argv) >= 2 and argv[0] == "tune" and not argv[1].startswith("-") and argv[1] not in SUBCOMMANDS:
        return [argv[0], "auto", *argv[1:]]
    return argv

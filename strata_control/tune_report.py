"""The account of a tune that the person can read: what was measured, what was tried, what was left out, and what to do next.

Everything here is derived from the result the tuner already returns (`autotune.TuneResult.to_dict()`); nothing is measured and no
number is invented. A missing figure is left out of the text, never shown as zero. The same text goes to the Terminal after a tune,
into a saved report that `strata tune report` prints again, and (as Markdown) into a `[Benchmark]` issue.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

SCHEMA = "strata.tune-report/1"

#: How an arm's status reads to a person.
_ARM_VERDICT = {"won": "faster", "tested": "no real gain", "dropped": "slower", "failed": "did not run", "skipped": "not run", "pending": "not run"}


def reports_dir() -> Path:
    from strata_control import presets
    return presets.preset_dir() / "reports"


def _pct(ratio: Any) -> str | None:
    if not isinstance(ratio, (int, float)):
        return None
    value = round(100 * (float(ratio) - 1.0), 1)
    return f"{value + 0.0:+.1f}%"            # + 0.0 turns -0.0 into 0.0


def _tok(value: Any) -> str | None:
    return f"{float(value):.1f} tok/s" if isinstance(value, (int, float)) and value else None


def arm_rows(r: dict[str, Any]) -> list[dict[str, str]]:
    """One row per setting the tuner measured or tried to: its name, the verdict in words, and the detail behind it."""
    rows = []
    for a in r.get("arms") or []:
        status = "won" if a.get("won") else a.get("status", "pending")
        change = _pct(a.get("ratio_decode"))
        if a.get("pairs"):
            detail = f"{change} against your settings, faster in {a.get('faster', 0)} of {a['pairs']} rounds" if change else ""
            if a.get("reason") and not a.get("won"):
                detail = a["reason"]
        else:
            detail = a.get("reason") or ""
        rows.append({"id": a.get("id", ""), "label": a.get("label", ""), "verdict": _ARM_VERDICT.get(status, status),
                     "status": status, "change": change or "", "detail": detail})
    return rows


def unlockable(r: dict[str, Any]) -> dict[str, list[str]]:
    """Settings left out only because they may change answers: what the person would tick to have them tried."""
    out: dict[str, list[str]] = {}
    for s in r.get("skipped") or []:
        kind = s.get("kind")
        if kind in ("placement", "nonrepeatable"):
            out.setdefault(kind, []).append(s.get("label", s.get("id", "")))
    return out


def next_step(r: dict[str, Any], preset_name: str | None = None) -> str:
    """The one thing to do now, for every way a tune can end."""
    code, reason = r.get("code") or "", r.get("reason_code") or ""
    if code == "faster":
        who = f"'{preset_name}'" if preset_name else "The preset"
        return (f"{who} is applied at every start. See why each setting was chosen with 'strata presets show', "
                "or go back with 'strata presets undo'.")
    if code == "already_fastest":
        return "Nothing to change. Tune again after an engine or driver update, or when you change the model or the context."
    if code == "busy_pc":
        return {"noisy": "Close other programs (browsers, games, other models) and tune again; the PC was too noisy to tell settings apart.",
                "drift": "Tune again when nothing else is running; the PC changed speed while measuring.",
                "not_tried": "Free some memory, or give it more time with --time, and tune again."}.get(
                    reason, "Tune again when the PC is idle.")
    if code == "failed":
        return {"no_memory": "Close other programs, or lower the context, or raise --vram-reserve-mib, and tune again.",
                "start_failed": "Check that the model starts with 'strata launch', then tune again. The engine log of the run says why it did not.",
                "tuner_error": "Nothing was changed. The run's logs were kept (see the notes); please include them if you report this."}.get(
                    reason, "Nothing was changed. Check the engine log of the run.")
    if code == "stopped":
        return "Nothing was changed. Run the tune again when you are ready."
    return ""


def render(r: dict[str, Any], *, preset_name: str | None = None) -> str:
    """The report as plain text, for the Terminal."""
    lines: list[str] = []
    values = r.get("values") or {}
    base = r.get("base") or {}
    noise = r.get("noise") or {}

    if base.get("decode_tok_s"):
        parts = [_tok(base["decode_tok_s"]) + " answers"]
        if base.get("pp_tok_s"):
            parts.append(f"{float(base['pp_tok_s']):.0f} tok/s prompts")
        runs = noise.get("base_runs") or []
        lines.append("Your settings: " + ", ".join(parts) + (f" (measured {len(runs)} times: " + ", ".join(f"{x:g}" for x in runs) + ")" if len(runs) > 1 else ""))
    if noise.get("threshold_pct") is not None:
        spread = noise.get("spread")
        lines.append(f"To count, a setting had to be at least {noise['threshold_pct']:+.1f}% faster"
                     + (f"; identical runs on this PC differed by {100 * spread:.1f}%." if isinstance(spread, (int, float)) else "."))

    rows = arm_rows(r)
    if rows:
        lines += ["", "What was tried:"]
        width = max(len(x["label"]) for x in rows)
        for x in rows:
            lines.append(f"  {x['label']:<{width}}  {x['verdict']:<13} {x['detail']}".rstrip())
    pr = r.get("per_request") or []
    if pr:
        lines += ["", "Settings that need no restart:"]
        from strata_control.tune_knobs import PER_REQUEST
        names = {row["key"]: row["label"] for row in PER_REQUEST}
        for p in pr:
            change = _pct(p.get("ratio"))
            lines.append(f"  {names.get(p.get('key'), p.get('key'))} {p.get('value')}: "
                         + ("faster" if p.get("won") else "no real gain") + (f" ({change})" if change else ""))

    skipped = [s for s in r.get("skipped") or [] if s.get("kind") != "time"]
    timed = [s for s in r.get("skipped") or [] if s.get("kind") == "time"]
    if skipped or timed:
        lines += ["", "Not tried:"]
        # the stored reasons say "tick ..." (the app has a tick box); in the Terminal the hint below names the flag instead
        short = {"placement": "can change answers slightly", "nonrepeatable": "answers would not repeat run to run"}
        for s in skipped:
            lines.append(f"  {s.get('label') or s.get('id')}: {short.get(s.get('kind'), s.get('reason'))}")
        if timed:
            lines.append(f"  {len(timed)} more, to stay inside the time you chose (use --mode thorough or --time to try them)")
    un = unlockable(r)
    if un.get("placement"):
        lines.append(f"  -> {len(un['placement'])} could be tried if you allow slightly different answers (--allow-small-differences).")
    if un.get("nonrepeatable"):
        lines.append(f"  -> {len(un['nonrepeatable'])} could be tried if you allow answers that do not repeat (--allow-not-repeatable).")

    step = next_step(r, preset_name or values.get("preset_name"))
    if step:
        lines += ["", "Next: " + step]
    return "\n".join(lines)


def markdown(r: dict[str, Any], *, title: str = "Tuning report") -> str:
    """The same account as Markdown, to paste into a [Benchmark] issue. It carries no paths, keys or user names."""
    values = r.get("values") or {}
    out = [f"## {title}", ""]
    out.append(f"- Result: {r.get('message') or r.get('outcome')}")
    out.append(f"- Mode: {r.get('mode')}; {values.get('minutes')} min of measuring; {r.get('runs')} engine runs")
    base = r.get("base") or {}
    chosen = r.get("chosen") or {}
    if base.get("decode_tok_s"):
        out.append(f"- Answers: {base['decode_tok_s']} tok/s before"
                   + (f", {chosen['decode_tok_s']} tok/s after" if chosen.get("decode_tok_s") and r.get("outcome") == "improved" else ""))
    noise = r.get("noise") or {}
    if r.get("classes"):
        out.append("- Machine class (the tuner's own grouping): " + "+".join(r["classes"]))
    if noise.get("threshold_pct") is not None:
        out.append(f"- A setting had to be {noise['threshold_pct']:+.1f}% faster to count; identical runs differed by "
                   f"{100 * noise.get('spread', 0):.1f}%")
    settings = (r.get("settings") or {})
    if settings.get("args") or settings.get("env") or settings.get("per_request"):
        out += ["", "Kept settings:", "", "```text"]
        for k, v in {**(settings.get("args") or {}), **(settings.get("env") or {}), **(settings.get("per_request") or {})}.items():
            out.append(f"{k} {'' if v is True else v}".rstrip())
        out.append("```")
    rows = arm_rows(r)
    if rows:
        out += ["", "| Setting | Verdict | Detail |", "| --- | --- | --- |"]
        out += [f"| {x['label']} | {x['verdict']} | {x['detail']} |" for x in rows]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- the saved report

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "tune"


def save(r: dict[str, Any], *, target: str, model: str | None = None, when: float | None = None) -> Path | None:
    """Keep the result and its readable account where `strata tune report` finds them. Returns None when it cannot be written:
    a tune must not fail because its report could not be saved. The run-by-run `trials` (each run's full argument list, with
    the model paths) are left out; the file is private to the user (0600) all the same."""
    when = when if when is not None else time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when))
    path = reports_dir() / f"{stamp}-{_slug(model or target)}.json"
    r = {k: v for k, v in r.items() if k != "trials"}
    payload = {"schema": SCHEMA, "at": when, "target": target, "model": model, "result": r,
               "text": render(r), "markdown": markdown(r, title=f"Tuning report: {model or target}")}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str, indent=1)
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def list_reports() -> list[Path]:
    """Saved reports, newest first."""
    try:
        return sorted(reports_dir().glob("*.json"), reverse=True)
    except OSError:
        return []


def load(ref: str | None = None) -> tuple[Path, dict[str, Any]]:
    """The newest report, or the one whose file name contains ``ref`` (or a path to a report file)."""
    if ref and Path(ref).is_file():
        path = Path(ref)
    else:
        found = [p for p in list_reports() if not ref or ref in p.name]
        if not found:
            raise FileNotFoundError("no saved tuning report" + (f" matches '{ref}'" if ref else " yet"))
        path = found[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("that is not a Strata tuning report")
    return path, data

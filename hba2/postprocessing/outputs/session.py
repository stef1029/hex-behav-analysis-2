"""
``session.json`` — everything the rig recorded about the setup, kept whole.

Mouse weight, protocol name, every protocol parameter, camera details and the ephys
block, rather than a one-line summary of them. Keeping the document whole costs 4 KB and
means nothing has to be recovered from the raw folder later.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._result import OutputResult


def _iso(moment) -> str | None:
    """A datetime as ISO text, or None when there is none."""
    if moment:
        return moment.isoformat()
    return None


def _stillness(stillness_path, scored_readings):
    """The rig's stillness record, or a statement that there was none, as a dict.

    Always present as an object rather than omitted when absent, so a reader
    can ask ``session["stillness"]["scored"]`` without first checking the key
    exists, and an old session answers the question honestly instead of
    looking like a newer one that failed.

    ``model.digest`` is the field to group sessions by: name and version are
    only as reliable as whoever remembered to change them, while the digest is
    computed from what the model actually does.
    """
    record = {"scored": bool(scored_readings)}

    if stillness_path is None:
        record["model"] = None
        record["note"] = (
            "no stillness metadata from the rig; either it predates stillness "
            "or the model was not running"
        )
        return record

    try:
        record.update(json.loads(Path(stillness_path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        # Not fatal: the readings and their scores are already written, and a
        # session is more useful with its data and no provenance than refused.
        record["model"] = None
        record["note"] = f"stillness metadata unreadable ({e})"
        return record

    used = (record.get("config_used") or {}).get("still_threshold")
    recommended = (record.get("model") or {}).get("recommended_threshold")
    if used is not None and recommended is not None and used != recommended:
        record["threshold_differs_from_model"] = {
            "used": used, "model_recommends": recommended,
        }
    return record


def write(out_dir, meta, daq, camera, rows, block_record, roles, layout,
          stillness_path=None, scored_readings=False):
    """Write the session metadata document.

    Args:
        roles: channel name -> what it is for, as the pipeline classified it.
        layout: the session's folder layout, as the cohort detected it.
        stillness_path: the rig's ``_scales_stillness.json``, if it wrote one.
        scored_readings: whether the scales CSV actually carried stillness
            columns. Recorded separately from the sidecar's existence: either
            can be present without the other, and conflating them would hide
            a half-recorded session.
    """
    path = Path(out_dir) / "session.json"

    if daq.used_legacy_names:
        channel_names = "translated_from_1_based"
    else:
        channel_names = "as_recorded"

    # Which blocks ran and what they actually did. The trials table carries the
    # block name per trial; this is what that name meant.
    blocks = [
        {
            "name": b.name,
            "display_name": b.display_name,
            "is_warmup": b.is_warmup,
            "entered_at": _iso(b.entered_at),
            "params": b.params,
        }
        for b in block_record
    ]

    # Which stillness model scored this session, kept whole like the rest.
    # A session is scored by exactly one model -- the scales server builds
    # the estimator once at startup -- so this is session-level, and the
    # per-reading columns in scales.csv need no model tag of their own.
    stillness = _stillness(stillness_path, scored_readings)

    document = {
        "session_id": meta.session_id,
        "layout_version": meta.layout_version,
        "mouse_id": meta.mouse_id,
        "mouse_weight": meta.mouse_weight,
        "start_time": _iso(meta.start_time),
        "protocol_name": meta.protocol_name,
        "phase": meta.phase,
        "requested_trials": meta.requested_trials,

        "rig": {
            "name": meta.rig_name,
            "number": meta.rig_number,
            "camera_serial": meta.camera_serial,
        },

        # From Tracker_data.json — the camera's own record of the recording.
        "camera": {
            "fps": camera.frame_rate,
            "width": camera.width,
            "height": camera.height,
            "pixel_format": camera.pixel_format,
            "n_frames": camera.n_frames,
            "n_dropped": camera.n_dropped,
            "start_time": _iso(camera.start_time),
            "end_time": _iso(camera.end_time),
        },

        # t0_wall makes relative times convertible to wall-clock without doing
        # float arithmetic on Unix epochs.
        "clock": {
            "name": "daq",
            "t0_wall": _iso(meta.start_time),
            "duration_s": daq.duration,
            "n_messages": daq.n_messages,
            "reliability": daq.reliability,
            "n_error_messages": daq.n_error_messages,
        },

        # What each recorded channel was for, so the session says so itself rather
        # than depending on a reader knowing the rule.
        "channels": dict(sorted(roles.items())),

        # What this session was read from. A processed session states its own
        # provenance rather than leaving it to be inferred from the raw folder,
        # which may later be moved or archived.
        "source": {
            "layout": layout,
            "layout_version_recorded": meta.layout_version,
            "channel_names": channel_names,
            "channels_renamed": daq.renamed_channels,
            "trials_columns_absent": rows.absent_columns,
            "trials_columns_renamed": rows.renamed_columns,
            "blocks_recorded": block_record.recorded,
        },

        "blocks": blocks,
        "blocks_recorded": block_record.recorded,

        "protocol_parameters": meta.protocol_parameters,
        "laser": meta.laser_parameters or None,
        "ephys": meta.ephys,

        "stillness": stillness,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return OutputResult(name="session", path=path, n_rows=1)

# Full-match video workflow

The published video is generated in two separate stages: a policy-controlled
simulation records a complete match trace, then an offline renderer turns that
trace into presentation footage. This separation keeps the match result
independent of camera composition and encoding.

## 1. Record a complete match

Use `tools/record_verified_policy_trace.py` to record one uninterrupted
1,600-step Stage-D rollout. A publishable trace must include:

- a score beginning at zero and ending at the reported final value;
- robot root pose, articulation joints, and mechanism state;
- positions and orientations for all 456 FUEL bodies;
- match clock, phase, HUB state, score, collection count, and terminal state;
- the selected checkpoint, environment template, seeds, and their checksums.

`tools/verified_topdown_trace.py` validates these invariants. Incomplete,
early-terminal, non-finite, or checksum-mismatched traces are rejected.

## 2. Validate visible motion

Before rendering the full match, inspect frames from the start, midpoint, late
match, and finish. Confirm that the robot, mechanisms, game pieces, camera
insets, and score all change consistently. Historical score summaries or
low-resolution onboard frames cannot reconstruct a new field-camera view
because they do not contain the complete world state.

## 3. Render from recorded state

Use `tools/render_verified_trace_topdown.py` with the approved camera state.
For each frame, the renderer restores the recorded robot and FUEL transforms,
updates render-visible state, and captures one frame without executing policy
actions or advancing simulation physics. The public composition uses a bright
slanted field view, three robot-camera insets, and live match telemetry.

Small QA samples should be rendered before the full 160-second pass. The
intake inset must remain outside the mechanism envelope and show the field
rather than the inside of the chamber.

## 4. Encode and verify

`tools/encode_rendered_video.sh` converts the lossless intermediate to H.264.
The publication file uses 1920 x 1080 video, 10 fps, 1,600 frames, `yuv420p`,
MP4 fast-start, and no audio. Validation checks the complete decode, frame
count, duration, resolution, codec, and final score.

The high-bitrate master is kept outside Git because of hosting limits. The
smaller publication encode is uploaded as a native GitHub attachment so it can
play directly on the project page.

## 5. Publish only public evidence

The repository stores the compact video, poster, source code, and a sanitized
checksum record. Checkpoints, raw traces, temporary AVI files, server paths,
failed attempts, operational logs, and QA artifacts remain outside the public
Git history.

The published score-201 record is in
[`media/score201-public-provenance.json`](media/score201-public-provenance.json).

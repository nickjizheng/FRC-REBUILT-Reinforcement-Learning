# Optional narrated overview plan

The repository now includes a genuine, uninterrupted
[160-second score-201 match](../README.md#demo) with an exact
[public provenance record](media/score201-public-provenance.json).
The shorter narrated overview remains optional. This plan defines its evidence
and shot order so it can be produced without mixing live training telemetry
with fixed-checkpoint results.

## 90-second storyboard

| Time | Visual | Narration / evidence |
|---:|---|---|
| 0–8 s | Title and full field | Camera-based full-match robot learning in a physics-faithful FRC REBUILT simulator. |
| 8–20 s | Overhead field and robot | Show HUBS, FUEL, BUMPS, TRENCHES, match timing, and legal scoring. |
| 20–32 s | Three synchronized onboard cameras | Explain what the actor can observe; distinguish actor inputs from critic-only training state. |
| 32–45 s | Intake, storage, turret, and shooting | Demonstrate that scoring comes through simulated mechanisms and game rules. |
| 45–60 s | Collectors, replay, learner, and checkpoint flow | Show distributed training and immutable checkpoint custody. |
| 60–74 s | One uninterrupted collect–return–score cycle | Use a deterministic checkpoint and display its short SHA and seed. |
| 74–84 s | Fixed-evaluation results | Show the +9.20 paired-mean promotion and the later −11.34 rejected candidate. |
| 84–90 s | Reproducibility card | Link the repository, evaluation protocol, results JSON, tests, and license. |

## Recording contract

- Capture at 1920 × 1080, 30 fps, with the simulator at policy speed 1.0.
- Use a named immutable checkpoint and record its SHA-256 and evaluation seed.
- Keep the actor's observations visually distinct from critic-only state.
- Include at least one failure or rejected result; do not show only the best
  rollout.
- Do not record terminals, credentials, private paths, hostnames, or unrelated
  desktop windows.
- Caption scores as fixed-evaluation results only when they can be traced to a
  checked report and exact checkpoint hash.

## Publishing

Upload the compact H.264 MP4 as a GitHub user attachment so the README receives
the native inline player, while retaining the checksum-pinned repository copy
for reproducibility. Keep the script, checkpoint SHA, seed, capture date, and
video checksum beside the public documentation.

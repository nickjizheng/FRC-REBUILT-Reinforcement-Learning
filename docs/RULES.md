# 2026 REBUILT rules model

`src/xrc_rebuilt/rules.py` is the simulator's single source of truth for the
official match clock, HUB state, FUEL staging, and the main field/game-piece
dimensions. SI values are derived from the source-unit dimensions in the
manual. The official CAD—not the rounded manual dimensions—remains the source
of truth for detailed collision meshes.

## Match clock and HUB state

An official match has 160 seconds of play:

| Elapsed simulation time | Arena clock | Segment | HUB state |
|---:|---:|---|---|
| 0–20 s | AUTO 0:20–0:00 | AUTO | both active |
| 20–30 s | TELEOP 2:20–2:10 | transition | both active |
| 30–55 s | 2:10–1:45 | shift 1 | alternating |
| 55–80 s | 1:45–1:20 | shift 2 | alternating |
| 80–105 s | 1:20–0:55 | shift 3 | alternating |
| 105–130 s | 0:55–0:30 | shift 4 | alternating |
| 130–160 s | 0:30–0:00 | end game | both active |

The alliance that scores more FUEL in AUTO has its own HUB inactive in shift
1; the states alternate for every following shift. On an AUTO tie, FMS randomly
selects the order. `select_first_inactive_alliance(..., seed=...)` preserves
that random behavior while making RL episodes reproducible.

HUB lighting uses the four exact Lit/Unlit bar meshes extracted from xRC
`level74`, at their original positions and colors. During a MATCH, active HUBS
show their ALLIANCE color at full brightness; the HUB that will rest first gets
an ALLIANCE-color/white chase during the TRANSITION SHIFT; lights pulse for the
3 seconds before a deactivation (and before the final buzzer); inactive HUBS
are off; and both HUBS turn white for the 3-second post-MATCH assessment period.
FUEL assessment continues for up to 3 seconds after AUTO, after TELEOP, and
after a HUB deactivates. `fuel_score_is_eligible` implements the active/inactive
and post-deactivation processing window; AUTO reward accounting must still
retain FUEL that entered during AUTO and is assessed by t=23 s as AUTO FUEL.

Source: FIRST, [2026 REBUILT Game Manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm), sections 5.4 (HUB lighting), 6.4 (MATCH periods and HUB status), and 6.5 (scoring assessment). FIRST's [2026 FMS game-data documentation](https://docs.wpilib.org/en/stable/docs/yearly-overview/2026-game-data.html) independently documents that the higher AUTO scorer is inactive first and that tie selection is random.

## FUEL staging and physics

The official nominal total is 504 FUEL:

- 24 in each of two DEPOTS (48),
- 24 in each of two OUTPOST chutes (48),
- up to 8 in each of six ROBOTS (0–48), and
- the remainder in the neutral zone (408 with no preloads; 360 with all 48).

The manual allows the neutral-zone count to fluctuate by approximately ±24
and allows an increase to as many as 600 total at District Championship and/or
FIRST Championship events. The simulator's standard episode uses the nominal
504 and all 48 legal preloads unless its scenario overrides that choice.

Official FUEL is a 5.91 in (15.0 cm) diameter high-density foam ball weighing
0.448–0.500 lb (approximately 0.203–0.227 kg). Physics uses the midpoint mass
unless domain randomization samples the official range.

Source: FIRST, [2026 REBUILT Game Manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm), sections 5.10.1 (FUEL) and 6.3.4 (SCORING ELEMENT staging).

## Field geometry

The official nominal carpet boundary is 651.2 by 317.7 in (16.54048 by
8.06958 m). It contains two HUBS, four BUMPS, four TRENCHES, two DEPOTS, two
OUTPOSTS, and two TOWERS. Key constants in the module include:

- HUB: 47 by 47 in footprint, centered 158.6 in from its ALLIANCE WALL; 41.7 in
  hexagonal opening with its front edge 72 in above carpet; four return exits.
- BUMP: 73.0 by 44.4 by 6.513 in, with 15-degree ramps.
- TRENCH: 65.65 by 47.0 by 40.25 in, with a 50.34 by 22.25 in clear opening.
- 32 AprilTags from the 36h11 family, each 8.125 in square.

Source: FIRST, [2026 REBUILT Game Manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm), sections 5.2–5.11. The manual identifies the official CAD as the exact geometric representation.

## HUB return routing

The official rule is that processed FUEL is randomly distributed into the
neutral zone through one of four HUB exits. FIRST does not publish an internal
travel-time distribution. For xRC fidelity, this simulator therefore uses the
project calibration target of 1.32 seconds mean travel time, bounded to the
official 3-second processing window. `sample_hub_routing_delay` uses a bounded
triangular distribution on [0, 3] seconds with mode 0.96 seconds, whose expected
mean is exactly 1.32 seconds. `sample_hub_exit` chooses all four exits uniformly.

The 1.32-second distribution is a simulator calibration, not an official FIRST
field dimension. The four exits and random neutral-zone routing are official;
see section 5.4 of the manual cited above.

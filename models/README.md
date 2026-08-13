# Selected checkpoints

This directory is the local checkpoint bank for the single canonical project
tree. Checkpoint binaries are intentionally ignored by Git, while
`SELECTED_MODELS.tsv` records their exact SHA-256 identities and roles.

The active bank contains one selected checkpoint for Stage A, Stage B, and
Stage D. Stage C has one formal champion plus the later Stage-D prefix source;
the latter is retained because it is an input to the final full-match lineage,
not as a competing Stage-C result. The score-201 interpolation is retained only
as the exact policy used for the published demonstration.

Superseded experiment checkpoints remain only in the sealed offline archive
on drive D and are not part of the canonical resumable project state.

Before using a checkpoint, verify it against `SELECTED_MODELS.tsv`, for example:

```powershell
Get-FileHash -Algorithm SHA256 models/selected/stageD_fsg9_3230_promoted_62bef28.pt
```

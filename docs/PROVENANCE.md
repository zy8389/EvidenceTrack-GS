# Source provenance

This full-source repository was assembled from:

- upstream repository: `CPy255/GeoTrack-GS`
- pinned upstream commit: `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`
- EvidenceTrack-GS package commit: `40ffc1f9ea47a5c57c3957423b48aff304f771f7`
- preserved package tag: `reproducible-package-2026-09-17`
- cumulative patch SHA-256: `a05aee6929d95e1eb00997df489de058bbe07429a175688c4c3627df0e687c8a`

The machine-readable `.research_revision_v2.json` repeats the pinned upstream commit. Runtime gates validate that provenance field and separately record the current materialized-repository commit; the latter is expected to differ from the upstream commit.

The assembly order was fixed as cumulative pre-AST patch, source overlay, then guarded AST integration. The cumulative patch explicitly deletes `evidence_track/geometry/adaptive_weighting.py`; that file must remain absent unless a future research change deliberately reintroduces and validates it.

`phase2_1_integrity.patch` is retained here for provenance. Day-to-day development should modify the materialized source files in the repository root directly. The earlier package layout remains recoverable from the Git tag above.

The assembly marker says validation was not run by the assembler itself. Post-assembly validation was run separately; the corresponding report and logs are under `docs/validation/`.

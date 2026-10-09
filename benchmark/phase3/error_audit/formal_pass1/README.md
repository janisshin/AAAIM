# Formal Pass 1 provenance

This directory separates immutable human-review exports from later analysis.

- `raw/<export-timestamp>/` contains byte-for-byte copies of the browser exports. Never edit or overwrite a raw snapshot; add a new timestamped snapshot instead.
- `provenance/<export-timestamp>.json` records file hashes and validation results for the corresponding raw snapshot.
- Future JSON/CSV comparison products belong under `derived/`, not under `raw/`.
- Formal Pass 2 exports and adjudications must remain outside this directory in separately named Pass 2/adjudication locations.

## Snapshots

- `20261001T214551Z` is preserved as received, but it is an incomplete checkpoint: 58 of 60 cases are marked complete. It cannot unlock Formal Pass 2.
- `20261002T072656Z` is the completed 60-case Pass 1 export. It passes the repository's complete-export validation and is eligible to unlock Formal Pass 2.

See each snapshot's provenance record for non-judgment validation details and hashes.

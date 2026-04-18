# Changelog

## [0.1.0] — 2026-04-16

First public release.

### Features
- Ratatui TUI with 4 panels: GPU stats / active jobs / history / log
- Nickel-based config (`config.ncl`) with JSON fallback
- Connector trait with `gemma` / `deepl` / `claude` implementations
- Runtime connector switch (`c` key)
- Job cancellation (`x` key) — kills child process
- tui-input for editable fields
- Progress parsing from subprocess stderr (`N/M (X%)` pattern)
- History persistence (XDG data dir)
- Help overlay (`?`)
- SVG screenshot generator (`cargo run --bin screenshot`)

### Smoke test
`cargo run --bin smoke config.ncl` runs config → connector registry → one translate roundtrip → GPU poll.

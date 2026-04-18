# FAQ

## Config file not found
Default looks for `config.ncl` in cwd. Specify path:
```bash
./target/release/translate-dashboard path/to/config.ncl
```
If you don't have `nickel-lang-cli`, the app falls back to `config.ncl.json` or `config.json` (plain JSON) at the same path.

## GPU panel shows no data
The `gpu.host` in config must be SSH-reachable (passwordless). Test:
```bash
ssh <gpu.host> nvidia-smi
```

## Can't cancel a running job
`x` key cancels the selected job (sends kill to the spawned subprocess). Long-running HTTP requests in-flight are not interrupted — the model call on the server finishes, but no further work runs.

## Connector switch during running job
Switch takes effect for jobs dispatched AFTER the switch. In-flight jobs keep the previous connector (by design).

## History gone after restart
Persisted to `~/.local/share/translate-dashboard/history.json` — if that's empty, check `$XDG_DATA_HOME`.

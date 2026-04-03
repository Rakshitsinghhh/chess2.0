# Play Against the Model

Quick entry point for playing against this chess neural network.

For full details, see:

- `play_vs_user_nn/README.md`

## Quick run (UI, drag-and-drop)

```bash
cd /home/rakshit/projects1/chess2.0
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn_ui.py
```

## Quick run (CLI)

```bash
cd /home/rakshit/projects1/chess2.0
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn.py
```

## Notes

- Default model path: `outputs/models/latest.pt`
- UI uses `pygame`; CLI does not require it
- PGNs are saved to `outputs/pgn/` (CLI mode)


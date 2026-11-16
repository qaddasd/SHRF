"""Exercise the full archive, consolidation and recovery lifecycle."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from shrf import SHRFConfig, SHRFLayer
from shrf.tape import IDTape


@torch.no_grad()
def main():
    torch.manual_seed(0)
    layer = SHRFLayer(SHRFConfig(d_model=32, depth=4, r_key=4, d_value=8)).eval()
    with TemporaryDirectory() as work:
        path = Path(work) / "example.sqlite"
        with IDTape(path) as tape:
            state = layer.init_state(1, tape=tape)
            x = torch.randn(1, 256, 32)
            layer(x, state)
            old_bytes = state.extra.numel() * state.extra.element_size()
            layer.consolidate(state, budget=4)
            new_bytes = state.extra.numel() * state.extra.element_size()
            expected = layer.W_v(x)[0, 1:]
        with IDTape(path) as tape:
            recovered = tape.recover_time(1, 256)
            torch.testing.assert_close(recovered.value, expected, rtol=0, atol=0)
            print(f"Recovered {len(recovered.time)} exact projected writes after reopening the archive.")
            print(f"Sketch tensor {old_bytes:,} -> {new_bytes:,} bytes, archive on disk {path.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()

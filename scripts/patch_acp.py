#!/usr/bin/env python3

from pathlib import Path
import sys


NEEDLE = (
    "assert self.owner.timeline.quantum_manager.formalism "
    "== BELL_DIAGONAL_STATE_FORMALISM, \\"
)


def patch_file(path: Path) -> None:
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    if "QuantumManagerBellDiagonal" in text and "isinstance(self.owner.timeline.quantum_manager, QuantumManagerBellDiagonal)" in text:
        print(f"{path}: already patched")
        return

    out = []
    changed = False

    for line in lines:
        if NEEDLE in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(
                f"{indent}from sequence.kernel.quantum_manager "
                f"import QuantumManagerBellDiagonal\n"
            )
            out.append(
                f"{indent}assert isinstance("
                f"self.owner.timeline.quantum_manager, "
                f"QuantumManagerBellDiagonal), \\\n"
            )
            changed = True
        else:
            out.append(line)

    if not changed:
        raise RuntimeError(f"{path}: did not find target assert")

    path.write_text("".join(out))
    print(f"{path}: patched")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/patch_acp.py /path/to/acp")
        raise SystemExit(2)

    acp_dir = Path(sys.argv[1]).resolve()
    patch_file(acp_dir / "generation.py")
    patch_file(acp_dir / "purification.py")


if __name__ == "__main__":
    main()
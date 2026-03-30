from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


N_SIM = 1000
OUT_DIR = Path("dataset5")
SLIM_SCRIPT = Path("slim5.slim")
EXTRACT_SCRIPT = Path("/Users/ekaterina/Desktop/RESEARCH/SLiM/output15.py")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )


def main() -> None:
    if not SLIM_SCRIPT.exists():
        raise FileNotFoundError(f"SLiM script not found: {SLIM_SCRIPT}")
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f"Extractor script not found: {EXTRACT_SCRIPT}")

    records: list[dict[str, object]] = []
    success_count = 0
    attempt_count = 0

    progress = tqdm(total=N_SIM, desc="Successful simulations")

    while success_count < N_SIM:
        attempt_count += 1
        sim_id = f"sim_{success_count}"

        sel_coeff = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.05))))
        admix_prop = 0.01
        neutrality_times = [0, 50, 100, 200, 500, 750, 1000, 1300, 1600, 1900]
        neutrality_time = int(np.random.choice(neutrality_times))

        trees_file = OUT_DIR / f"{sim_id}.trees"
        csv_file = OUT_DIR / f"{sim_id}.csv"

        if trees_file.exists():
            trees_file.unlink()
        if csv_file.exists():
            csv_file.unlink()

        try:
            slim_cmd = [
                "slim",
                "-d", f"SEL_COEFF={sel_coeff}",
                "-d", f"ADMIX_PROP={admix_prop}",
                "-d", f"NEUTRALITY_PERIOD={neutrality_time}",
                "-d", f'OUTFILE="{trees_file.as_posix()}"',
                str(SLIM_SCRIPT),
            ]
            run_command(slim_cmd)

            if not trees_file.exists():
                raise RuntimeError(f"SLiM finished but did not create {trees_file}")

            extract_cmd = [
                sys.executable,
                str(EXTRACT_SCRIPT),
                str(trees_file),
                str(csv_file),
            ]
            run_command(extract_cmd)

            if not csv_file.exists():
                raise RuntimeError(f"Feature extractor finished but did not create {csv_file}")

            records.append(
                {
                    "sim_id": sim_id,
                    "file": csv_file.name,
                    "trees_file": trees_file.name,
                    "neutrality_time": neutrality_time,
                    "sel_coeff": sel_coeff,
                    "admix_prop": admix_prop,
                    "attempt": attempt_count,
                }
            )

            success_count += 1
            progress.update(1)

        except Exception as e:
            if trees_file.exists():
                trees_file.unlink()
            if csv_file.exists():
                csv_file.unlink()
            print(f"[attempt {attempt_count}] failed: {e}")

    progress.close()

    labels = pd.DataFrame(records)
    labels.to_csv(OUT_DIR / "labels.csv", index=False)

    print(f"Dataset generation complete: {success_count} successful simulations.")
    print(f"Saved labels to: {OUT_DIR / 'labels.csv'}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.superposition.asf_online_superposition import (  # noqa: E402
    ASFOnlineSuperpositionManager,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize ASF checkpoints from a superposition bundle."
    )
    parser.add_argument("--bundle", type=str, required=True,
                        help="Path to a *.bundle.pt file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where materialized checkpoints will be written.")
    parser.add_argument("--scenes", type=str, nargs="+", required=True,
                        help="Scene names to materialize, e.g. seq1 seq58")
    parser.add_argument("--approximate", action="store_true",
                        help="Use pure unbinding from storage_state * context_table instead of exact scene_deltas.")
    return parser.parse_args()


def max_abs_diff_state_dict(a_state, b_state):
    max_diff = 0.0
    max_key = ""
    for key in a_state.keys():
        a_val = a_state[key]
        b_val = b_state[key]
        if not torch.is_tensor(a_val) or not torch.is_tensor(b_val):
            continue
        if a_val.shape != b_val.shape:
            continue
        diff = (a_val.detach().cpu() - b_val.detach().cpu()).abs()
        if diff.numel() == 0:
            continue
        cur_max = float(diff.max())
        if cur_max > max_diff:
            max_diff = cur_max
            max_key = key
    return max_diff, max_key


def main():
    args = parse_args()

    bundle_path = os.path.abspath(args.bundle)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manager = ASFOnlineSuperpositionManager.from_bundle(bundle_path, output_dir=output_dir)
    exact_flag = not bool(args.approximate)
    mode_name = "exact" if exact_flag else "approximate"

    print(f"* Bundle = {bundle_path}", flush=True)
    print(f"* Output dir = {output_dir}", flush=True)
    print(f"* Materialization mode = {mode_name}", flush=True)
    print(f"* Scenes = {args.scenes}", flush=True)

    summary_lines = [
        f"bundle={bundle_path}",
        f"mode={mode_name}",
    ]

    for scene_name in args.scenes:
        state_dict = manager.materialize_full_state(scene_name, exact=exact_flag)
        ckpt_path = os.path.join(output_dir, f"{scene_name}.checkpoint")
        save_checkpoint(ckpt_path, state_dict, {
            "scene": scene_name,
            "bundle_path": bundle_path,
            "materialization_mode": mode_name,
            "exact": exact_flag,
        })
        print(f"* Wrote {scene_name}: {ckpt_path}", flush=True)
        summary_lines.append(f"{scene_name}_checkpoint={ckpt_path}")

        if scene_name == manager.base_scene:
            summary_lines.append(f"{scene_name}_vs_exact_max_abs_diff=0.0")
            summary_lines.append(f"{scene_name}_vs_exact_max_abs_diff_key=")
            continue

        exact_state = manager.materialize_full_state(scene_name, exact=True)
        max_diff, max_key = max_abs_diff_state_dict(state_dict, exact_state)
        print(
            f"* {scene_name} vs exact: max_abs_diff={max_diff:.8g} key={max_key}",
            flush=True,
        )
        summary_lines.append(f"{scene_name}_vs_exact_max_abs_diff={max_diff}")
        summary_lines.append(f"{scene_name}_vs_exact_max_abs_diff_key={max_key}")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"* Summary = {summary_path}", flush=True)


if __name__ == "__main__":
    main()

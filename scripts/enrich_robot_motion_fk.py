import argparse
import pickle
from pathlib import Path

from general_motion_retargeting.motion_data import enrich_robot_motion_with_fk, resolve_torch_device


def main():
    parser = argparse.ArgumentParser(
        description="Add local_body_pos and link_body_list to a robot-motion PKL."
    )
    parser.add_argument("--input", required=True, help="Input robot-motion PKL.")
    parser.add_argument("--output", required=True, help="Output enriched robot-motion PKL.")
    parser.add_argument("--robot", default="unitree_g1", help="Robot key used for FK.")
    parser.add_argument(
        "--device",
        default="auto",
        help="FK device: auto, cpu, cuda, cuda:0, etc.",
    )
    args = parser.parse_args()

    with Path(args.input).open("rb") as file:
        motion_data = pickle.load(file)
    enriched_motion = enrich_robot_motion_with_fk(
        motion_data,
        robot=args.robot,
        device=args.device,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(enriched_motion, file)
    print(
        f"Saved {output_path} with {len(enriched_motion['root_pos'])} frames, "
        f"{enriched_motion['local_body_pos'].shape[1]} bodies, "
        f"device={resolve_torch_device(args.device)}"
    )


if __name__ == "__main__":
    main()

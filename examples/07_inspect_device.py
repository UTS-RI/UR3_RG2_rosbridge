"""List connected devices, every stream profile and all standard sensor controls."""

import argparse
from pathlib import Path

from realsense_lab import inspect_devices, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Save the full report as JSON")
    args = parser.parse_args()
    reports = inspect_devices()
    if not reports:
        raise SystemExit("No RealSense device found. Check USB access and the README setup notes.")
    for report in reports:
        print(f"\n{report['info'].get('name', 'RealSense')} | firmware {report['info'].get('firmware_version', '?')}")
        for sensor in report["sensors"]:
            streams = sorted({f"{p['stream']}:{p['index']}" for p in sensor["profiles"]})
            print(f"  {sensor['name']}: {', '.join(streams)}")
            print(f"    {len(sensor['profiles'])} profiles; exposure ROI: {sensor['exposure_roi']}")
            for name, option in sensor["options"].items():
                print(f"    {name}: {option.get('value', 'unavailable')} "
                      f"[{option.get('min', '?')}..{option.get('max', '?')}] "
                      f"{'read-only' if option.get('read_only') else ''}")
    if args.save:
        save_json(args.save, reports)
        print(f"Full profiles and options saved to {args.save}")


if __name__ == "__main__":
    main()

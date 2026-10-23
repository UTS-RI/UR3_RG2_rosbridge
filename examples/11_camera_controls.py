"""Inspect controls, or explicitly set an option/auto-exposure ROI."""

import argparse
import json
from pathlib import Path

from realsense_lab import RealSenseCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", default="depth", help="depth, colour, motion or an exact sensor name")
    parser.add_argument("--set", nargs=2, metavar=("OPTION", "VALUE"), action="append", default=[])
    parser.add_argument("--roi", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--advanced-json", type=Path, help="Export existing D400 advanced settings")
    parser.add_argument("--serial")
    args = parser.parse_args()
    with RealSenseCamera(serial=args.serial) as camera:
        controls = camera.controls
        for name, value in args.set:
            controls.set_option(args.sensor, name, float(value))
            print(f"{name} = {controls.get_option(args.sensor, name)}")
        if args.roi:
            controls.set_roi(args.sensor, *args.roi)
            print(f"Exposure ROI: {controls.get_roi(args.sensor)}")
        print(json.dumps(controls.options(args.sensor), indent=2))
        if args.advanced_json:
            settings = controls.advanced_settings_json()
            args.advanced_json.parent.mkdir(parents=True, exist_ok=True)
            args.advanced_json.write_text(settings, encoding="utf-8")
            print(f"Advanced settings saved to {args.advanced_json}")


if __name__ == "__main__":
    main()

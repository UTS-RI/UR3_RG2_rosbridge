# RealSense Lab

Capture colour images, depth and 3D points from an **Intel RealSense D435i** with
Python. Start with an example, then use the functions in `src/realsense_lab/`
for a project. Camera features are checked when requested.

For UR3 robot control, follow [Returning to the robot project](#returning-to-the-robot-project).

## 1. Set up

Use **Python 3.12**. Run these commands from the project folder.

Create a virtual environment on **Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Or on **Linux or macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project with camera support:

```bash
python -m pip install -r requirements.txt
```

Windows and Linux need a compatible
[RealSense Python package](https://pypi.org/project/pyrealsense2/#files).
Linux may also need [camera access rules](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md).
For macOS, camera support needs [additional SDK setup](https://github.com/realsenseai/librealsense/blob/master/doc/installation_osx.md).
Installing Python packages alone may not be enough.

**Without a camera**, install the base package instead:

```bash
python -m pip install -e .
python examples/01_rgb_depth.py --demo
```

The demo creates artificial images using NumPy and Matplotlib.

## 2. Try the camera

Connect the D435i to a USB 3 port and close other camera applications.

```bash
python -m realsense_lab.check --camera
python examples/01_rgb_depth.py
```

The example shows RGB, a depth heatmap and an overlay. In the heatmap, nearby
surfaces are red and distant surfaces are blue.

Set the overlay strength with `--alpha`, from `0` for RGB to `1` for full depth
colours. The default is `0.5`. Pixels without depth keep their RGB colour.
This option works in examples 01, 04, 05 and 13.

```bash
python examples/04_live_rgb_depth.py --alpha 0.3
```

Try the two point cloud views:

```bash
python examples/02_pointcloud_heatmap.py
python examples/03_pointcloud_rgb.py
```

A point cloud is a set of 3D positions. Example 02 colours each point by depth.
Example 03 uses the RGB image colours. Drag the plot to rotate the view.

These examples also accept `--demo`. Saving is optional.
To save an image without opening a window:

```bash
python examples/01_rgb_depth.py --demo --save output/rgb_depth.png
```

## 3. Start coding

```python
from realsense_lab import RealSenseCamera, depth_heatmap, make_pointcloud

with RealSenseCamera() as camera:
    frame = camera.capture()

rgb = frame.native_rgb
depth = frame.depth_m
heatmap = depth_heatmap(depth)
cloud_rgb = make_pointcloud(frame, colour="rgb")
cloud_depth = make_pointcloud(frame, colour="depth")

print("Image size:", rgb.shape)
print("Number of 3D points:", len(cloud_rgb.points))
```

The `with` block starts the camera and releases it afterwards. The captured data
remain available. To practise without a camera, import and use `DemoCamera`
instead of `RealSenseCamera`.

**Three things to remember:**

1. `frame.native_rgb` is the original colour image, with RGB values from 0 to 255.
   Use `frame.rgb` when colour pixels need to match the depth image.
2. `frame.depth_m` stores depth in metres along the camera's forward axis.
   Use `frame.valid_depth` to exclude missing or invalid measurements.
3. Image pixels use `[y, x]` indexing. Point clouds use XYZ coordinates in metres,
   with X pointing right, Y down and Z forwards.

### Why RGB and depth can look different

The RGB panel shows the original colour image. The depth sensor sees a wider area,
so objects can appear at different positions in the two panels.

The overlay and point clouds use aligned colour data. By default,
`alignment="depth"` maps RGB onto the depth view. That aligned image can have black
borders and gaps where colour coverage or depth measurements are missing.

Use `RealSenseCamera(alignment="colour")` to map depth onto the full RGB view.
The aligned images share pixel positions. Some pixels can still lack depth.
The original sensor images are available as `frame.native_rgb` and
`frame.native_depth_raw`. Their pixel positions do not directly match.

## 4. Choose an example

Run any example from the project folder. Add `--help` to see its options.
Examples marked **demo** also work with `--demo`.

1. [RGB and depth](examples/01_rgb_depth.py): colour image, heatmap and overlay. **Demo**.
2. [Depth point cloud](examples/02_pointcloud_heatmap.py): 3D points coloured by depth. **Demo**.
3. [RGB point cloud](examples/03_pointcloud_rgb.py): 3D points with image colours. **Demo**.
4. [Live images](examples/04_live_rgb_depth.py): continuously update RGB and depth. **Demo**.
5. [Save data](examples/05_save_data.py): export images, measurements and point clouds. **Demo**.
6. [Custom processing](examples/06_own_processing.py): remove a background and measure depth. **Demo**.
7. [Inspect the camera](examples/07_inspect_device.py): list sensors, resolutions and settings.
8. [Infrared](examples/08_infrared.py): show the left and right greyscale images. **Demo**.
9. [Motion sensors](examples/09_imu.py): read acceleration and angular velocity, with optional CSV export. **Experimental. Demo available**.
10. [Depth filters](examples/10_depth_filters.py): compare original and filtered depth.
11. [Camera controls](examples/11_camera_controls.py): inspect or change supported settings.
12. [Record](examples/12_record.py): save camera streams to a `.db3` file.
13. [Playback](examples/13_playback.py): process a recording without a camera.
14. [Calibration and measurement](examples/14_calibration_measurement.py): inspect calibration and measure a 3D distance. **Demo**.
15. [Live point cloud](examples/15_live_pointcloud.py): update a 3D view with RGB or depth colours. **Demo**.

Close a live view with `Q`, `Escape` or the window close button.
If the live point cloud is slow, try `--step 16` to display fewer points.

**Saving data:** Examples do not save files by default. Add `--save` with a path
to save an image, measurements or a recording:

```bash
python examples/05_save_data.py --demo --save output/captures
python examples/09_imu.py --demo --save output/imu.csv
python examples/12_record.py --save output/session.db3
```

Example 05 creates a new capture folder. `rgb.png` contains the original colour image.
PNG files are pictures. NPZ files retain
measurements and can be loaded with `load_frame()`. PLY files contain 3D points and
colours. Example 14 saves calibration JSON alongside the image when `--save` is used.
Example 11 uses `--advanced-json FILE` to export advanced settings.

**Motion sensors (experimental):** Live IMU capture still needs validation on the
lab computer. The current attempt fails because Linux blocks sensor access with
`scan_element ... Permission denied`. The demo uses synthetic data and does not
verify the hardware. This status also applies when IMU is enabled for video or recording.

The IMU contains an accelerometer and a gyroscope.
Acceleration includes gravity. Orientation angles need additional processing.
Use example 09 to collect individual motion samples. `frame.motion` only contains
the samples attached to a video capture.

**Recording:** The camera installation includes SDK 2.59.1 or later.
The examples use `.db3` recordings. Add `--all-streams` to example 12 to include
both infrared images and both motion sensors.

## 5. If something does not work

**Camera or motion sensor access fails on Linux:** Install the official
[RealSense access rules](https://github.com/realsenseai/librealsense/blob/master/scripts/setup_udev_rules.sh)
and reconnect the camera. This requires administrator access. RGB and depth can
work even when motion sensor access fails. The rules must include IMU/HID/IIO
access. Installing `pyrealsense2` with pip does not install these system rules.
After setup, test the motion sensors first:

```bash
python examples/09_imu.py --samples 100
```

Then check all streams together:

```bash
python -m realsense_lab.check --all-streams
```

**No compatible camera package:** Check the supported Python version and platform
in the package link above. The demo can run without the camera package.

**No image window:** Matplotlib needs a graphical backend. Ubuntu may need
`python3-tk`. Use `--save output/image.png` with example 01 to save an image instead.

**Camera cannot start:** Close other camera applications and check the USB
connection. Use example 07 to inspect supported settings. Some stream combinations
cannot run together.

**PowerShell blocks activation:** Use `.\.venv\Scripts\python.exe` in place of
`python` in the commands above.

## Project files

`examples/` contains the programs to run and adapt.
`src/realsense_lab/` contains the reusable camera and processing functions.
`tests/` contains automated checks that run without a camera.

To run the checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

IMU support remains experimental until live capture is verified. Interactive
windows also need checking on the lab computer.
The examples do not update firmware or change factory calibration.

## Returning to the robot project

The `main` branch contains the UR3 simulation, robot control examples and lab setup instructions.
Close any running camera examples and commit or stash local changes before switching.
Run the following command from the project folder.

```text
git switch main
```

Switching branches changes the project files and README. It does not change the active Python environment.
If the camera virtual environment is active, run `deactivate` in the same terminal.
Then open the [robot README](https://github.com/UTS-RI/UR3_RG2_rosbridge/tree/main#readme)
and follow its instructions to activate the existing `ur3` environment or create one.

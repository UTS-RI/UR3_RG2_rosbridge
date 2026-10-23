"""Control only the OnRobot RG2 gripper attached to the real robot."""

from pynput import keyboard
import onRobot.gripper as gripper


ROBOT_IP = "192.168.0.194"
GRIPPER_ID = 0

CLOSED_WIDTH_MM = 10.0
OPEN_WIDTH_MM = 80.0
GRIP_FORCE_N = 20.0


def setup_gripper() -> gripper.RG:
    """Create the RG2 interface and confirm communication with the gripper."""
    print("Connecting to RG2 gripper via onRobot library...")
    rg2 = gripper.RG(ROBOT_IP, GRIPPER_ID)
    initial_width = rg2.get_rg_width()
    print(f"RG2 gripper connected. Current width: {initial_width:.1f} mm")
    return rg2


def main() -> None:
    try:
        rg2 = setup_gripper()
    except Exception as exc:
        raise SystemExit(f"Failed to connect to RG2 gripper: {exc}") from exc

    print("\n--- Gripper Controls ---")
    print(f"Z   - Close to {CLOSED_WIDTH_MM:.0f} mm")
    print(f"X   - Open to {OPEN_WIDTH_MM:.0f} mm")
    print("ESC - Exit program\n")

    pressed_keys = set()

    def command_gripper(width_mm: float, action: str) -> None:
        print(f"Commanding gripper: {action} ({width_mm:.0f} mm)")
        try:
            rg2.rg_grip(width_mm, GRIP_FORCE_N)
            actual_width = rg2.get_rg_width()
            print(f"Gripper width: {actual_width:.1f} mm")
        except Exception as exc:
            print(f"Gripper error: {exc}")

    def on_press(key):
        try:
            key_char = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.esc:
                return False
            return None

        # Ignore keyboard auto-repeat while a control key remains held down.
        if key_char in pressed_keys:
            return None

        pressed_keys.add(key_char)
        if key_char == "z":
            command_gripper(CLOSED_WIDTH_MM, "CLOSE")
        elif key_char == "x":
            command_gripper(OPEN_WIDTH_MM, "OPEN")

        return None

    def on_release(key):
        try:
            pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    print("Gripper control stopped.")


if __name__ == "__main__":
    main()

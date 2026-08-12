import threading

from pynput import keyboard


class Keyboard:
    def __init__(self) -> None:
        self.LINEAR_SPEED = 0.14
        self.ANGULAR_SPEED = 3.14
        self.ARM_ROT_SPEED = 0.00628

        self.v_lin_x = 0.0
        self.v_lin_y = 0.0
        self.v_lin_z = 0.0
        self.v_yaw = 0.0
        self.v_arm_rot = 0.0

        self.key_states = {}
        self.key_states_lock = threading.Lock()

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.base_modes = {0: True}
        self.active_robot = 0
        self.active_arm_index = 0
        self.all_robot_arms = {0: [0]}
        self.num_robots = 1

    def on_press(self, key) -> None:
        try:
            with self.key_states_lock:
                self.key_states[key] = True
        except AttributeError:
            pass

    def on_release(self, key) -> None:
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[
                    self.active_robot
                ]  # toggle mobile base
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(
                    self.all_robot_arms[self.active_robot]
                )
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError:
            pass

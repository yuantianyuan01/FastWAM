import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

GRIPPER_LENGTH = 0.125


def gather_joint_info(model, data, joint_name_or_index):
    body_id = model.joint(joint_name_or_index).bodyid[0]
    # root_body_id = model.body(body_id).rootid[0]
    body_joint_qpos = model.joint(joint_name_or_index).qposadr[0]
    joint_range = model.joint(joint_name_or_index).range
    max_range = joint_range[1] if joint_range[1] != 0 else joint_range[0]

    joint_info = {
        "joint_axis": model.joint(joint_name_or_index).axis,
        "joint_position": model.joint(joint_name_or_index).pos,
        "joint_range": model.joint(joint_name_or_index).range,
        "joint_pos": data.qpos[body_joint_qpos],
        "joint_body_position": data.xpos[body_id],
        "joint_body_orientation": data.xmat[body_id].reshape(3, 3),  # Body orientation matrix
        "max_range": max_range,
        "joint_type": model.joint(joint_name_or_index).type,
        "joint_qpos_adr": body_joint_qpos,
        "joint_id": model.joint(joint_name_or_index).id,
    }
    return joint_info


def step_linear_path(
    to_handle_dist,
    current_pos,
    current_quat,
    step_size,
    is_reverse=False,
    gripper_length=0,
):
    path = {"mocap_pos": [], "mocap_quat": []}
    path["mocap_pos"].append(current_pos)
    path["mocap_quat"].append(current_quat)

    dist = np.linalg.norm(to_handle_dist)

    if is_reverse:
        dist += gripper_length
    else:
        dist -= gripper_length

    # path forward
    for _i in range(int(dist / step_size)):
        # in direction of to_handle_dist
        angle = np.arctan2(to_handle_dist[1], to_handle_dist[0])
        if not is_reverse:
            next_pos = current_pos + step_size * np.array([np.cos(angle), np.sin(angle), 0])
        else:
            next_pos = current_pos - step_size * np.array([np.cos(angle), np.sin(angle), 0])
        path["mocap_pos"].append(next_pos)
        path["mocap_quat"].append(current_quat)
        dist -= np.linalg.norm(next_pos - current_pos)
        current_pos = next_pos
    return path


def step_circular_path(
    current_pos,
    current_quat,
    joint_info,
    max_joint_angle,
    n_waypoints=10,
    gripper_length=0,
):
    """
    joint_info:
        joint_body_position
        joint_axis
        joint_body_orientation
        joint_position
        joint_range
        joint_pos
    """

    def rotation_matrix_from_axis_angle(axis, angle):
        """Create rotation matrix from axis and angle using scipy's reliable implementation"""
        axis = axis / np.linalg.norm(axis)
        # Use scipy's implementation which is more reliable

        R_matrix = R.from_rotvec(axis * angle).as_matrix()
        return R_matrix

    ## extract joint info
    joint_body_position = joint_info["joint_body_position"]
    joint_axis_local = joint_info["joint_axis"]

    # Convert joint axis from local body frame to global frame
    # Get the body's orientation matrix
    body_orientation = joint_info["joint_body_orientation"]
    joint_axis = body_orientation @ joint_axis_local

    # joint position is in the joint frame, so we need to convert it to the world frame
    # joint_position = joint_info["joint_position"] + need to convert to world frame by multiplying body orientation
    joint_position = body_orientation @ joint_info["joint_position"] + joint_body_position

    # Use gripper position to find the arc that gripper follows
    handle_position = current_pos
    handle_orientation = current_quat

    # get offset from joint to gripper
    handle_offset = handle_position - joint_position

    if np.abs(joint_axis[2]) > 0.9:
        # if rotating along the global z axis, make the height same
        joint_position[2] = handle_position[2]

        # For Z-axis rotation, we need to ensure the rotation is in the XY plane
        # The gripper offset should be in the XY plane only
        handle_offset_xy = handle_offset.copy()
        handle_offset_xy[2] = 0  # Zero out Z component for XY plane rotation
        handle_offset = handle_offset_xy
    current_joint_angle = joint_info["joint_pos"]

    # Calculate relative angles (change from current to max)
    angle_change = max_joint_angle - current_joint_angle
    angles = np.linspace(0, angle_change, n_waypoints + 1)
    if np.abs(angle_change) < 0.1:
        angles = np.linspace(0, -max_joint_angle, n_waypoints + 1)

    # Get gripper orientation matrix
    gripper_orientation_matrix = R.from_quat(handle_orientation, scalar_first=True).as_matrix()

    # Calculate the finger center offset from joint (this follows the circular arc)
    # The finger center should be at the handle position initially
    finger_center_offset_from_joint = handle_position - joint_position

    if np.abs(joint_axis[2]) > 0.9:
        # For Z-axis rotation, ensure the finger center offset is in XY plane only
        finger_center_offset_from_joint_xy = finger_center_offset_from_joint.copy()
        finger_center_offset_from_joint_xy[2] = 0
        finger_center_offset_from_joint = finger_center_offset_from_joint_xy

    path = {"mocap_pos": [], "mocap_quat": []}

    for angle in angles:
        # Calculate rotation matrix for this angle
        R_matrix = rotation_matrix_from_axis_angle(joint_axis, angle)

        # Rotate finger center offset around the joint (finger center follows circular arc)
        rotated_finger_center_offset = R_matrix @ finger_center_offset_from_joint

        # Calculate new finger center position (this follows the circular arc)
        next_finger_center_pos = joint_position + rotated_finger_center_offset

        # Calculate new gripper orientation by applying the joint rotation to the original orientation
        next_gripper_orientation_matrix = R_matrix @ gripper_orientation_matrix
        next_gripper_z_axis = next_gripper_orientation_matrix[:, 2]  # Z-axis points towards handle

        # Calculate gripper base position by offsetting from finger center
        # Gripper base is offset by half gripper length in the negative Z direction (away from handle)
        gripper_base_offset_from_finger = -0.5 * gripper_length * next_gripper_z_axis
        next_gripper_base_pos = next_finger_center_pos + gripper_base_offset_from_finger

        # Convert to quaternion
        next_quat = R.from_matrix(next_gripper_orientation_matrix).as_quat(scalar_first=True)

        # Use gripper base position for the mocap trajectory (but finger center follows the arc)
        path["mocap_pos"].append(next_gripper_base_pos)
        path["mocap_quat"].append(next_quat)

    visualize = False
    if visualize:
        visualize_path(path, joint_position=joint_position)

    return path


def visualize_path(
    path,
    title="Gripper Base Path Visualization",
    save_path=None,
    joint_position=None,
    show_finger_center=True,
):
    """
    Comprehensive visualization of the gripper base path and finger center arc.

    Args:
        path: Dictionary with 'mocap_pos' and 'mocap_quat' lists (representing gripper base positions)
        title: Title for the plot
        save_path: Optional path to save the plot
        joint_position: Optional joint position to visualize
        show_finger_center: If True, also show the finger center arc for reference
    """
    if not path or "mocap_pos" not in path or len(path["mocap_pos"]) == 0:
        print("No path data to visualize")
        return

    # Convert to numpy arrays for easier manipulation
    # Handle case where positions might be tuples or lists
    if len(path["mocap_pos"]) > 0:
        # Convert each position to numpy array if it isn't already
        positions = np.array(
            [np.array(pos) if not isinstance(pos, np.ndarray) else pos for pos in path["mocap_pos"]]
        )
    else:
        positions = np.array([])

    # Handle quaternions similarly
    if "mocap_quat" in path and len(path["mocap_quat"]) > 0:
        quaternions = np.array(
            [
                np.array(quat) if not isinstance(quat, np.ndarray) else quat
                for quat in path["mocap_quat"]
            ]
        )
    else:
        quaternions = None

    print(f"Visualizing gripper base path with {len(positions)} points")
    print(
        f"Base position range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
        f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
        f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]"
    )

    # Calculate finger center positions for reference (if quaternions available)
    finger_center_positions = None
    if show_finger_center and quaternions is not None and len(quaternions) > 0:
        finger_center_positions = []
        for _i, (base_pos, quat) in enumerate(zip(positions, quaternions)):
            # Calculate finger center position by offsetting from base
            rot_matrix = R.from_quat(quat, scalar_first=True).as_matrix()
            gripper_z_axis = rot_matrix[:, 2]  # Z-axis points towards handle
            finger_offset = 0.5 * GRIPPER_LENGTH * gripper_z_axis
            finger_pos = base_pos + finger_offset
            finger_center_positions.append(finger_pos)
        finger_center_positions = np.array(finger_center_positions)
        print(
            f"Finger center position range: X[{finger_center_positions[:, 0].min():.3f}, {finger_center_positions[:, 0].max():.3f}], "
            f"Y[{finger_center_positions[:, 1].min():.3f}, {finger_center_positions[:, 1].max():.3f}], "
            f"Z[{finger_center_positions[:, 2].min():.3f}, {finger_center_positions[:, 2].max():.3f}]"
        )

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 2D Top-down view (X-Y plane)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        positions[:, 0],
        positions[:, 1],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Gripper Base Path",
    )
    ax1.scatter(
        positions[0, 0], positions[0, 1], color="green", s=100, label="Base Start", zorder=5
    )
    ax1.scatter(positions[-1, 0], positions[-1, 1], color="red", s=100, label="Base End", zorder=5)

    # Add finger center path if available
    if finger_center_positions is not None:
        ax1.plot(
            finger_center_positions[:, 0],
            finger_center_positions[:, 1],
            "r--s",
            linewidth=1,
            markersize=3,
            alpha=0.7,
            label="Finger Center Arc",
        )
        ax1.scatter(
            finger_center_positions[0, 0],
            finger_center_positions[0, 1],
            color="darkgreen",
            s=80,
            marker="s",
            label="Finger Start",
            zorder=5,
        )
        ax1.scatter(
            finger_center_positions[-1, 0],
            finger_center_positions[-1, 1],
            color="darkred",
            s=80,
            marker="s",
            label="Finger End",
            zorder=5,
        )

    # Add joint position if provided
    if joint_position is not None:
        ax1.scatter(
            joint_position[0],
            joint_position[1],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position (finger center for reference)
        if finger_center_positions is not None:
            ax1.plot(
                [joint_position[0], finger_center_positions[0, 0]],
                [joint_position[1], finger_center_positions[0, 1]],
                "k--",
                alpha=0.5,
                linewidth=1,
                label="Joint-Finger Offset",
            )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Top-down View (X-Y)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # 2D Side view (X-Z plane)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(positions[:, 0], positions[:, 2], "r-o", linewidth=2, markersize=4, label="Path")
    ax2.scatter(positions[0, 0], positions[0, 2], color="green", s=100, label="Start", zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="End", zorder=5)

    # Add joint position if provided
    if joint_position is not None:
        ax2.scatter(
            joint_position[0],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position
        ax2.plot(
            [joint_position[0], positions[0, 0]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Side View (X-Z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # 2D Front view (Y-Z plane)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(positions[:, 1], positions[:, 2], "g-o", linewidth=2, markersize=4, label="Path")
    ax3.scatter(positions[0, 1], positions[0, 2], color="green", s=100, label="Start", zorder=5)
    ax3.scatter(positions[-1, 1], positions[-1, 2], color="red", s=100, label="End", zorder=5)

    # Add joint position if provided
    if joint_position is not None:
        ax3.scatter(
            joint_position[1],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position
        ax3.plot(
            [joint_position[1], positions[0, 1]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Front View (Y-Z)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # 3D view
    ax4 = plt.subplot(2, 2, 4, projection="3d")
    ax4.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Path",
    )
    ax4.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        color="green",
        s=100,
        label="Start",
        zorder=5,
    )
    ax4.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="red",
        s=100,
        label="End",
        zorder=5,
    )

    # Add joint position if provided
    if joint_position is not None:
        ax4.scatter(
            joint_position[0],
            joint_position[1],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position to show the offset
        ax4.plot(
            [joint_position[0], positions[0, 0]],
            [joint_position[1], positions[0, 1]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.set_title("3D View")

    # Add legend for orientation arrows
    from matplotlib.patches import FancyArrowPatch

    legend_elements = [
        FancyArrowPatch((0, 0), (0.1, 0), color="red", linewidth=2, label="Gripper X-axis"),
        FancyArrowPatch((0, 0), (0.1, 0), color="green", linewidth=2, label="Gripper Y-axis"),
        FancyArrowPatch((0, 0), (0.1, 0), color="blue", linewidth=2, label="Gripper Z-axis"),
    ]
    ax4.legend(handles=legend_elements, loc="upper right")

    # Add orientation arrows if quaternions are available
    if quaternions is not None and len(quaternions) > 0:
        # Show orientation at regular intervals along the path
        n_arrows = min(8, len(positions))  # Show up to 8 arrows
        arrow_indices = np.linspace(0, len(positions) - 1, n_arrows, dtype=int)

        for i, idx in enumerate(arrow_indices):
            if idx < len(quaternions):
                # Convert quaternion to rotation matrix
                rot_matrix = R.from_quat(quaternions[idx], scalar_first=True).as_matrix()

                # Create arrows representing gripper orientation (X, Y, Z axes)
                arrow_length = 0.15
                origin = positions[idx]

                # X-axis (red), Y-axis (green), Z-axis (blue) of gripper
                axes = [
                    (rot_matrix[:, 0], "red", "X"),  # X-axis
                    (rot_matrix[:, 1], "green", "Y"),  # Y-axis
                    (rot_matrix[:, 2], "blue", "Z"),  # Z-axis
                ]

                for axis_direction, color, _axis_name in axes:
                    ax4.quiver(
                        origin[0],
                        origin[1],
                        origin[2],
                        axis_direction[0],
                        axis_direction[1],
                        axis_direction[2],
                        length=arrow_length,
                        color=color,
                        alpha=0.8,
                        linewidth=2,
                    )

                # Add text label for the point
                ax4.text(
                    origin[0],
                    origin[1],
                    origin[2],
                    f"P{i}",
                    fontsize=8,
                    color="black",
                    weight="bold",
                )

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Path visualization saved to {save_path}")

    plt.show()

    # Print path statistics
    print("\nPath Statistics:")
    print("Gripper Base Path:")
    print(f"  Total distance: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.3f} m")
    print(f"  Number of waypoints: {len(positions)}")
    print(f"  Start position: {positions[0]}")
    print(f"  End position: {positions[-1]}")

    if finger_center_positions is not None:
        finger_distance = np.sum(np.linalg.norm(np.diff(finger_center_positions, axis=0), axis=1))
        print("Finger Center Arc:")
        print(f"  Total distance: {finger_distance:.3f} m")
        print(f"  Start position: {finger_center_positions[0]}")
        print(f"  End position: {finger_center_positions[-1]}")

    if joint_position is not None:
        print(f"Joint position: {joint_position}")
        print(
            f"Distance from joint to base start: {np.linalg.norm(positions[0] - joint_position):.3f} m"
        )
        print(
            f"Distance from joint to base end: {np.linalg.norm(positions[-1] - joint_position):.3f} m"
        )
        if finger_center_positions is not None:
            print(
                f"Distance from joint to finger start: {np.linalg.norm(finger_center_positions[0] - joint_position):.3f} m"
            )
            print(
                f"Distance from joint to finger end: {np.linalg.norm(finger_center_positions[-1] - joint_position):.3f} m"
            )

    # Print orientation statistics if available
    if quaternions is not None and len(quaternions) > 0:
        print(f"Start orientation (quaternion): {quaternions[0]}")
        print(f"End orientation (quaternion): {quaternions[-1]}")

        # Calculate orientation change
        start_rot = R.from_quat(quaternions[0], scalar_first=True)
        end_rot = R.from_quat(quaternions[-1], scalar_first=True)
        orientation_diff = start_rot.inv() * end_rot
        angle_diff = orientation_diff.magnitude()
        print(f"Total orientation change: {np.degrees(angle_diff):.1f}°")

    return fig

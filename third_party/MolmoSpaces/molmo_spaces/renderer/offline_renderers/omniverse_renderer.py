"""
Offline renderer for rendering scenes from saved MujocoState dataset

Omniverse RTX Renderer
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html

Kit/Issac Sim Min Requirements:
- 535.129.03 Linux (RTX 3070/Quadro)
- Intel i7 Gen5 or  AMD Ryzen
- 16GB/32GB RAM
- Ubuntu 20.04 or 22.04

"""

import os
from pathlib import Path
from typing import NoReturn

import tqdm

try:
    # Omniverse class must be instanitated before it can be imported
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(enagle_cameras=True, headless=True)
    simulation_app = app_launcher.app

    import carb.settings
    import omni
    import omni.isaac.core.utils.stage as stage_utils
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    # from pxr import Semantics


except ImportError:
    print("Omniverse is not installed")


# import mujoco
# from mujoco.usd import exporter


def load_state(self, usd_path: str = None) -> None:
    # open USD stage
    stage_utils.open_stage(usd_path)


def check_if_camera_valid(camera_path) -> bool:
    context = omni.usd.get_context()
    stage = context.get_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)

    if not camera_prim.IsValid():
        print(f"Camera at path {camera_path} is not valid")
        return False
    if camera_prim.GetTypeName() == "Camera":
        return True
    else:
        print(f"{camera_prim} is not a Camera type")
        return False


def check_if_camera_resolution_valid(camera_resolution: tuple[int, int]) -> None:
    pass


class ImageWriter(rep.Writer):
    def __init__(
        self,
        output_dir: str,
        image_format: str = "png",
        rgb: bool = True,
        normals: bool = False,
        semantic_segmentation: bool = True,
        frame_padding: int = 4,
    ) -> None:
        self.output_dir = output_dir
        if output_dir:
            self._backend = rep.BackendDispatch(output_dir=output_dir)
        self.write_ready = False
        self._frame_id = 0
        self._frame_padding = frame_padding
        self._sequence_id = 0
        self._image_format = image_format
        self.rgb = rgb
        self.normals = normals
        self.semantic_segmentation = semantic_segmentation

        self.annotators = []
        if rgb:
            self.annotators.append(rep.Annotator(rep.AnnotatorType.RGB))
        if normals:
            self.annotators.append(rep.Annotator(rep.AnnotatorType.NORMALS))
        if semantic_segmentation:
            self.annotators.append(rep.Annotator(rep.AnnotatorType.SEMANTIC_SEGMENTATION))

    def write(self, data: dict) -> None:
        """
        Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/render_dataset_with_omniverse.py#L440
        """
        if self.write_ready:
            for ann_name, ann_data in data["annotators"].items():
                for _idx, (camera_name, data) in enumerate(ann_data.items()):
                    file_name = (
                        Path(camera_name).stem
                        / f"{self._frame_id:0{self._frame_padding}d}.{self._image_format}"
                    )
                    if ann_name == "rgb":
                        filepath = Path(self.output_dir) / "rgb" / file_name
                        self._backend.write_image(filepath, data["data"])
                    elif ann_name == "normals":
                        raise NotImplementedError("Normals are not supported")
                    elif ann_name == "semantic_segmentation":
                        filepath = Path(self.output_dir) / "semantic_segmentation" / file_name
                        self._backend.write_image(filepath, data["data"])
            self._frame_id += 1


class OfflineOmniverseRenderer:
    """
    For photorealistic RGB rendering

    reference:
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/render_dataset_with_omniverse.py
    """

    def __init__(
        self,
        output_dir: str,
        camera_names: list[str],
        camera_resolution: list[tuple[int, int]],
        num_frames: int = 1000,
    ) -> None:
        self.rgb_writer = ImageWriter(output_dir=output_dir)
        self.num_frames = num_frames  # same as number of traj_len
        self.rt_subframes = 1
        self.initial_skip = 0
        self.camera_names = camera_names
        self.camera_resolution = camera_resolution

    @classmethod
    def from_usd_state_data_path(cls, usd_path: str, state_path: str, output_dir: str = None):
        load_state(usd_path)
        # read camera names and num_frames from render_metadata.txt
        render_metadata_path = Path(usd_path).parent / "render_metadata.txt"
        assert render_metadata_path.exists(), "render_metadata.txt does not exist"
        with open(render_metadata_path, "r") as f:
            metadata = {}
            for line in f.readlines():
                if ":" not in line:
                    continue
                key, value = line.strip().split(":", 1)
                metadata[key.strip()] = value.strip()

        camera_names = eval(
            metadata.get("camera_names", "[]")
        )  # Convert string representation to list
        camera_resolution = eval(
            metadata.get("camera_resolution", "[]")
        )  # Convert string representation to list of tuples
        num_frames = int(metadata.get("num_frames", "0"))
        if output_dir is None:
            output_dir = os.path.dirname(usd_path)
        return cls(
            output_dir=output_dir,
            camera_names=camera_names,
            camera_resolution=camera_resolution,
            num_frames=num_frames,
        )

    def reset(self) -> None:
        self.camera_resolution = None
        self.render_products = []
        self.num_frames = 0

    @property
    def camera_resolution(self):
        return self._camera_resolution  # (width, height)

    @camera_resolution.setter
    def camera_resolution(self, camera_resolution: tuple[int, int]) -> None:
        self._camera_resolution = camera_resolution

    @property
    def camera_names(self):
        return self._camera_names

    @camera_names.setter
    def camera_names(self, camera_names: list[str]) -> None:
        self._camera_names = camera_names

    @classmethod
    def from_path(cls, usd_path: str, output_dir: str = None):
        load_state(usd_path)
        if output_dir is None:
            output_dir = os.path.dirname(usd_path)
        return cls(output_dir=output_dir)

    def _init(self) -> bool:
        assert self.usd_path is not None, "usd_path is not set"
        self._load_state(self.usd_path)

        if carb.settings.get_settings().get("/omni/replicator/captureOnPlay"):
            carb.settings.get_settings().set_bool("/omni/replicator/captureOnPlay", False)

        carb.settings.get_settings().set_bool("/app/renderer/waitIdle", False)
        carb.settings.get_settings().set_bool("/app/hydraEngine/waitIdle", False)
        carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        carb.settings.get_settings().set("/rtx/pathtracing/spp", 30)
        carb.settings.get_settings().set_bool(
            "/exts/omni.replicator.core/Orchestrator/enabled", True
        )

        # create render products
        assert self.camera_names is not None, "camera_names is not set"
        assert self.camera_resolution is not None, "camera_resolution is not set"
        for camera_name in self.camera_names:
            camera_path = f"/Camera/{camera_name}"
            rp = rep.create.render_product(camera_path, self.camera_resolution, force_new=True)
            self.render_products.append(rp)

        # attach render products to the writer
        if self.render_products:
            self.rgb_writer.attach_render_products(self.render_products)
        else:
            print("No render products to attach")
            return False
        return True

    def start_rendering(self) -> None:
        # start rendering
        ## start_recorder()
        self._init()

        # run_recording_loop()
        for _ in range(self.initial_skip):
            rep.orchestrator.step(rt_subframes=1, delta_time=None, pause_timeline=False)

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_end_time(self.num_frames)

        with tqdm(total=self.num_frames) as pbar:
            for _ in range(self.num_frames):  # are we rendering one image at a time?
                timeline.forward_one_frame()
                rep.orchestrator.step(
                    rt_subframes=self.rt_subframes, delta_time=None, pause_timeline=True
                )
                pbar.update(1)

        # finish_recording()
        timeline.stop()
        rep.orchestrator.wait_until_complete()

        # clear_recorder()
        self.writer.detach()
        self.writer = None
        for rp in self.render_products:
            rp.destroy()
        self.render_products = []
        stage_utils.clear_state()
        stage_utils.update_stage()

        # while loop;
        # stage_utils.update_stage()

        # if saving video, process_folders()

    def save_rendering(self, output_dir: str) -> None:
        # save renderings
        pass

    def _update(self) -> None:
        pass

    def _load(self, usd_path: str) -> None:
        stage_utils.open_stage(usd_path)

    def _init_recorder(self, usd_path: str) -> None:
        # open USD stage
        self._load(usd_path)

    def add_light(self, pos, intensity, radius, color, obj_name, light_type) -> NoReturn:
        self.exporter.add_light(pos, intensity, radius, color, obj_name, light_type)
        raise NotImplementedError("add_light is not implemented for OfflineOmniverseRenderer")

    def add_camera(self, pos, rotation_xyz, obj_name) -> NoReturn:
        self.exporter.add_camera(pos, rotation_xyz, obj_name)
        raise NotImplementedError("add_camera is not implemented for OfflineOmniverseRenderer")


if __name__ == "__main__":
    offline_omniverse_renderer = OfflineOmniverseRenderer()
    offline_omniverse_renderer.start_rendering()

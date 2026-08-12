from typing import Any

import mujoco as mj
import numpy as np

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer


def prepare_locals_for_super(
    local_vars, args_name="args", kwargs_name="kwargs", ignore_kwargs=False
):
    assert args_name not in local_vars, f"`prepare_locals_for_super` does not support {args_name}."
    new_locals = {k: v for k, v in local_vars.items() if k != "self" and "__" not in k}
    if kwargs_name in new_locals:
        if ignore_kwargs:
            new_locals.pop(kwargs_name)
        else:
            kwargs = new_locals.pop(kwargs_name)
            kwargs.update(new_locals)
            new_locals = kwargs
    return new_locals


class MjFilamentRenderer(MjAbstractRenderer):
    def __init__(
        self,
        model_bindings: MjModelBindings = None,
        device_id: int | None = None,
        height: int = 720,
        width: int = 1280,
        max_geom: int = 10000,
        model: mj.MjModel | None = None,
        **kwargs: Any,
    ) -> None:
        assert model_bindings is not None or model is not None, (
            "model_bindings or model must be provided"
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self._width = width
        self._height = height

        if model_bindings is not None and model is not None:
            assert model_bindings.model == model, "model_bindings and model must be the same"
        model = model_bindings.model if model_bindings is not None else model
        self._model = model

        self._scene = mj.MjvScene(model=model, maxgeom=max_geom)
        self._scene_option = mj.MjvOption()

        # Turn off site rendering
        self._scene_option.sitegroup *= 0

        # Enable shadow rendering by default (shadows are controlled by lights with castshadow enabled)
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        self._mjr_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
        # mj.mjr_resizeOffscreen(width, height, self._mjr_context)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
        self._mjr_context.readDepthMap = mj.mjtDepthMap.mjDEPTH_ZEROFAR

        # Default render flags.
        self._depth_rendering = False
        self._segmentation_rendering = False

        # Track if textures need to be uploaded (set to True when textures are modified)
        # NOTE: We start with False because textures are loaded from model at MjrContext creation
        # We only need to upload if textures are modified AFTER renderer initialization
        self._textures_need_upload = False

    @property
    def scene(self) -> mj.MjvScene:
        return self._scene

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def enable_depth_rendering(self) -> None:
        self._segmentation_rendering = False
        self._depth_rendering = True

    def disable_depth_rendering(self) -> None:
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = True
        self._depth_rendering = False

    def disable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = False

    def geomid_to_bodyid(self, geomid):
        return self.model.geom_bodyid[geomid]

    def render(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        height = height or self._height
        width = width or self._width
        rect = mj.MjrRect(0, 0, width, height)

        original_flags = self._scene.flags.copy()

        # Enable shadow rendering (required for shadows to appear in rendered images)
        # Shadows are controlled by lights with castshadow enabled
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        # Using segmented rendering for depth makes the calculated depth more
        # accurate at far distances.
        if self._depth_rendering or self._segmentation_rendering:
            self._scene.flags[mj.mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mj.mjtRndFlag.mjRND_IDCOLOR] = True

        # Upload textures to GPU before rendering if textures have been modified
        # This is necessary when textures are modified via model.tex_data
        # Only upload when needed to avoid performance overhead
        if self._textures_need_upload:
            self.upload_textures()
            self._textures_need_upload = False

        if self._depth_rendering:
            out_shape = (rect.height, rect.width)
            out_dtype = np.float32
        else:
            out_shape = (rect.height, rect.width, 3)
            out_dtype = np.uint8

        if out is None:
            out = np.empty(out_shape, dtype=out_dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(
                    f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`"
                    " instead. When using depth rendering, the out array should be of"
                    " shape `(width, height)` and otherwise (width, height, 3)."
                    f" Got `(self.height, self.width)={(self.height, self.width)}` and"
                    f" `self._depth_rendering={self._depth_rendering}`."
                )

        # Render scene and read contents of RGB and depth buffers.
        mj.mjr_render(rect, self._scene, self._mjr_context)

        if self._depth_rendering:
            mj.mjr_readPixels(rgb=None, depth=out, viewport=rect, con=self._mjr_context)

            # Get the distances to the near and far clipping planes.
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent

            # Calculate OpenGL perspective matrix values in float32 precision
            # so they are close to what glFrustum returns
            # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
            zfar = np.float32(far)
            znear = np.float32(near)
            c_coef = -(zfar + znear) / (zfar - znear)
            d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

            # In reverse Z mode the perspective matrix is transformed by the following
            c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
            d_coef = np.float32(-0.5) * d_coef

            # We need 64 bits to convert Z from ndc to metric depth without noticeable
            # losses in precision
            out_64 = out.astype(np.float64)

            # Undo OpenGL projection
            # Note: We do not need to take action to convert from window coordinates
            # to normalized device coordinates because in reversed Z mode the mapping
            # is identity
            out_64 = d_coef / (out_64 + c_coef)

            # Cast result back to float32 for backwards compatibility
            # This has a small accuracy cost
            out[:] = out_64.astype(np.float32)

            # Reset scene flags.
            np.copyto(self._scene.flags, original_flags)
        elif self._segmentation_rendering:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

            # Convert 3-channel uint8 to 1-channel uint32.
            image3 = out.astype(np.uint32)
            segimage = image3[:, :, 0] + image3[:, :, 1] * (2**8) + image3[:, :, 2] * (2**16)
            # Remap segid to 3-channel (object ID, object type, body ID) triplet
            # Seg ID 0 is background -- will be remapped to (-1, -1, -1).

            # Find the maximum segment ID in the image to size the output array correctly
            max_segid = np.max(segimage) if segimage.size > 0 else 0

            # Create output array with size to accommodate all possible segment IDs
            # Add 1 to account for 0-based indexing and ensure we have enough space
            segid2output = np.full((max_segid + 1, 3), fill_value=-1, dtype=np.int32)

            visible_geoms = [g for g in self._scene.geoms[: self._scene.ngeom] if g.segid != -1]
            visible_segids = np.array([g.segid + 1 for g in visible_geoms], np.int32)
            visible_objid = np.array([g.objid for g in visible_geoms], np.int32)
            visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)
            visible_bodyid = np.array(
                [self.geomid_to_bodyid(g.objid) for g in visible_geoms], np.int32
            )

            # Only set values for valid segment IDs that are within bounds
            valid_mask = (visible_segids >= 0) & (visible_segids < segid2output.shape[0])
            if np.any(valid_mask):
                segid2output[visible_segids[valid_mask], 0] = visible_objid[valid_mask]
                segid2output[visible_segids[valid_mask], 1] = visible_objtype[valid_mask]
                segid2output[visible_segids[valid_mask], 2] = visible_bodyid[valid_mask]

            out = segid2output[segimage]

            # Reset scene flags.
            np.copyto(self._scene.flags, original_flags)
        else:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

        return out

    def render_rgb(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        height = height or self._height
        width = width or self._width
        rect = mj.MjrRect(0, 0, width, height)

        # Enable shadow rendering (required for shadows to appear in rendered images)
        # Shadows are controlled by lights with castshadow enabled
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        # Using segmented rendering for depth makes the calculated depth more
        # accurate at far distances.
        if self._depth_rendering or self._segmentation_rendering:
            self._scene.flags[mj.mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mj.mjtRndFlag.mjRND_IDCOLOR] = True

        # Upload textures to GPU before rendering if textures have been modified
        # This is necessary when textures are modified via model.tex_data
        # Only upload when needed to avoid performance overhead
        if self._textures_need_upload:
            self.upload_textures()
            self._textures_need_upload = False

        if self._depth_rendering:
            out_shape = (rect.height, rect.width)
            out_dtype = np.float32
        else:
            out_shape = (rect.height, rect.width, 3)
            out_dtype = np.uint8

        if out is None:
            out = np.empty(out_shape, dtype=out_dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(
                    f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`"
                    " instead. When using depth rendering, the out array should be of"
                    " shape `(width, height)` and otherwise (width, height, 3)."
                    f" Got `(self.height, self.width)={(self.height, self.width)}` and"
                    f" `self._depth_rendering={self._depth_rendering}`."
                )

        # Render scene and read contents of RGB and depth buffers.
        mj.mjr_render(rect, self._scene, self._mjr_context)

        if self._depth_rendering:
            mj.mjr_readPixels(rgb=None, depth=out, viewport=rect, con=self._mjr_context)
        elif self._segmentation_rendering:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
        else:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

        return out

    def upload_textures(self, data: mj.MjData | None = None) -> None:
        if self.model.ntex == 0:
            log.debug("upload_textures(): Skipping - no textures in model (ntex == 0)")
            return

        for tex_id in range(self.model.ntex):
            mj.mjr_uploadTexture(self.model, self._mjr_context, tex_id)

    def mark_textures_dirty(self) -> None:
        self._textures_need_upload = True

    def update(
        self,
        data: mj.MjData,
        camera: int | str | mj.MjvCamera = -1,
        scene_option: mj.MjvOption | None = None,
    ) -> None:
        if not isinstance(camera, mj.MjvCamera):
            camera_id = camera
            if isinstance(camera_id, str):
                camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA.value, camera_id)
                if camera_id == -1:
                    raise ValueError(f'The camera "{camera}" does not exist.')
            if camera_id < -1 or camera_id >= self.model.ncam:
                raise ValueError(
                    f"The camera id {camera_id} is out of range [-1, {self.model.ncam})."
                )

            # Render camera.
            camera = mj.MjvCamera()
            camera.fixedcamid = camera_id

            # Defaults to mjCAMERA_FREE, otherwise mjCAMERA_FIXED refers to a
            # camera explicitly defined in the model_bindings.
            if camera_id == -1:
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultFreeCamera(self.model, camera)
            else:
                camera.type = mj.mjtCamera.mjCAMERA_FIXED

        scene_option = scene_option or self._scene_option
        mj.mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )

    def close(self) -> None:
        if hasattr(self, "_mjr_context") and self._mjr_context:
            self._mjr_context.free()
        self._mjr_context = None


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")

    args = parser.parse_args()

    if args.model == "":
        print("Must provide a model via --model option")
        exit(1)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Given model '{args.model}' is not a valid file")
        exit(1)

    model = mj.MjModel.from_xml_path(model_path.as_posix())
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    renderer = MjFilamentRenderer(model=model)
    renderer.update(data=data)

    image = renderer.render()
    pil_image = Image.fromarray(image)
    pil_image.save("test_render.png")

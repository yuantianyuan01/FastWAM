import sys
from queue import Queue
from typing import Any, Literal

import numpy as np
from mujoco import (
    MjData,
    MjModel,
    MjrContext,
    MjrRect,
    MjvCamera,
    MjvOption,
    MjvScene,
    mj_name2id,
    mjr_readPixels,
    mjr_render,
    mjr_resizeOffscreen,
    mjr_setBuffer,
    mjr_uploadTexture,
    mjtCamera,
    mjtCatBit,
    mjtDepthMap,
    mjtFontScale,
    mjtFramebuffer,
    mjtObj,
    mjtRndFlag,
    mjv_defaultFreeCamera,
    mjv_updateScene,
)

from molmo_spaces.env.mj_extensions import MjModelBindings

# from molmo_spaces.env.vector_env import MuJoCoVectorEnv # -- this import creates lots of problem. why?
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer, MultithreadRenderer


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


class MjOpenGLRenderer(MjAbstractRenderer):
    """Renders MuJoCo scenes with OpenGL."""

    def __init__(
        self,
        model_bindings: MjModelBindings = None,
        device_id: int | None = None,
        height: int = 720,
        width: int = 1280,
        max_geom: int = 10000,
        model: MjModel | None = None,
        **kwargs: Any,
    ) -> None:
        assert model_bindings is not None or model is not None, (
            "model_bindings or model must be provided"
        )
        """Initializes a new `Renderer`.

        Args:
          model: an mujoco.Mjmodel instance.
          device_id: The index of the device to use for rendering.
          height: image height in pixels.
          width: image width in pixels.
          max_geom: Optional integer specifying the maximum number of geoms that can
            be rendered in the same scene. If None this will be chosen automatically
            based on the estimated maximum number of renderable geoms in the model_bindings.

        Raises:
          ValueError: If `camera_id` is outside the valid range, or if `width` or
            `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
        """
        if device_id is None:
            try:
                import torch

                if torch.cuda.is_available():
                    device_id = 0
            except ImportError:
                pass

        super().__init__(**prepare_locals_for_super(locals()))

        self._width = width
        self._height = height

        if model_bindings is not None and model is not None:
            assert model_bindings.model == model, "model_bindings and model must be the same"
        model = model_bindings.model if model_bindings is not None else model
        self._model = model

        self._scene = MjvScene(model=model, maxgeom=max_geom)
        self._scene_option = MjvOption()

        # Turn off site rendering
        self._scene_option.sitegroup *= 0

        # Enable shadow rendering by default (shadows are controlled by lights with castshadow enabled)
        self._scene.flags[mjtRndFlag.mjRND_SHADOW] = True

        # Create render contexts.
        # TODO(nimrod): Figure out why pytype doesn't like gl_context.GLContext
        self._context_is_cgl = False
        if device_id is None:
            from mujoco import gl_context

            self._gl_context = gl_context.GLContext(width, height)  # type: ignore
            self._context_is_cgl = sys.platform == "darwin"
        else:
            from molmo_spaces.renderer.opengl_context import EGLGLContext

            self._gl_context = EGLGLContext(width, height, device_id)
        self._gl_context.make_current()
        self._mjr_context = MjrContext(model, mjtFontScale.mjFONTSCALE_150.value)
        mjr_resizeOffscreen(width, height, self._mjr_context)
        mjr_setBuffer(mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
        self._mjr_context.readDepthMap = mjtDepthMap.mjDEPTH_ZEROFAR

        # TODO In MacOS, keeping the context locked seems to preclude others to progress,
        #  so it doesn't look like we can achieve true parallelism through multi threading?
        #  This also happens at the end of render()
        if self._context_is_cgl:
            from mujoco.cgl import cgl

            cgl.CGLUnlockContext(self._gl_context._context)

        # Default render flags.
        self._depth_rendering = False
        self._segmentation_rendering = False

        # Track if textures need to be uploaded (set to True when textures are modified)
        # NOTE: We start with False because textures are loaded from model at MjrContext creation
        # We only need to upload if textures are modified AFTER renderer initialization
        self._textures_need_upload = False

    @property
    def scene(self) -> MjvScene:
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
        """Renders the scene as a numpy array of pixel values.

        Args:
          out: Alternative output array in which to place the resulting pixels. It
            must have the same shape as the expected output but the type will be
            cast if necessary. The expted shape depends on the value of
            `self._depth_rendering`: when `True`, we expect `out.shape == (width,
            height)`, and `out.shape == (width, height, 3)` when `False`.

        Returns:
          A new numpy array holding the pixels with shape `(H, W)` or `(H, W, 3)`,
          depending on the value of `self._depth_rendering` unless
          `out is None`, in which case a reference to `out` is returned.

        Raises:
          RuntimeError: if this method is called after the close method.
        """

        height = height or self._height
        width = width or self._width
        rect = MjrRect(0, 0, width, height)

        original_flags = self._scene.flags.copy()

        # Enable shadow rendering (required for shadows to appear in rendered images)
        # Shadows are controlled by lights with castshadow enabled
        self._scene.flags[mjtRndFlag.mjRND_SHADOW] = True

        # Using segmented rendering for depth makes the calculated depth more
        # accurate at far distances.
        if self._depth_rendering or self._segmentation_rendering:
            self._scene.flags[mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mjtRndFlag.mjRND_IDCOLOR] = True

        if self._gl_context is None:
            raise RuntimeError("render cannot be called after close.")

        self._gl_context.make_current()

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
        mjr_render(rect, self._scene, self._mjr_context)

        if self._depth_rendering:
            mjr_readPixels(rgb=None, depth=out, viewport=rect, con=self._mjr_context)

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
            mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

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
            mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

        out[:] = np.flipud(out)

        # TODO In MacOS, keeping the context locked seems to preclude others to progress,
        #  so it doesn't look like we can achieve true parallelism through multi threading?
        #  This also happens at the end of __init__()
        if self._context_is_cgl:
            from mujoco.cgl import cgl

            cgl.CGLUnlockContext(self._gl_context._context)

        return out

    def upload_textures(self, data: MjData | None = None) -> None:
        """Upload all textures to the GPU render context.

        This should be called after modifying texture data in model.tex_data
        to ensure the changes are visible in rendered images.

        NOTE: This only uploads textures to THIS renderer's context (MjOpenGLRenderer).
        The passive viewer has its own separate renderer context and won't see these updates.

        Args:
            data: Optional MjData to use for updating the scene after texture upload.
                  If provided, will call mjv_updateScene() to refresh the scene.
        """
        import logging

        log = logging.getLogger(__name__)

        if self._gl_context is None or self._mjr_context is None:
            log.debug("upload_textures(): Skipping - GL context or Mjr context is None")
            return

        # Skip if no textures exist
        if self.model.ntex == 0:
            log.debug("upload_textures(): Skipping - no textures in model (ntex == 0)")
            return

        log.debug(f"upload_textures(): Uploading {self.model.ntex} textures to GPU render context")
        self._gl_context.make_current()
        # Upload all textures to the render context
        for tex_id in range(self.model.ntex):
            mjr_uploadTexture(self.model, self._mjr_context, tex_id)

        # Unlock context if needed (for macOS)
        if self._context_is_cgl:
            from mujoco.cgl import cgl

            cgl.CGLUnlockContext(self._gl_context._context)

    def mark_textures_dirty(self) -> None:
        """Mark that textures have been modified and need to be uploaded.

        Call this after modifying texture data in model.tex_data to ensure
        the changes will be uploaded before the next render.
        """
        self._textures_need_upload = True

    def update(
        self,
        data: MjData,
        camera: int | str | MjvCamera = -1,
        scene_option: MjvOption | None = None,
    ) -> None:
        """Updates geometry used for rendering.

        Args:
          data: An instance of `MjData`.
          camera: An instance of `MjvCamera`, a string or an integer
          scene_option: A custom `MjvOption` instance to use to render
            the scene instead of the default.

        Raises:
          ValueError: If `camera_id` is outside the valid range, or if camera does
            not exist.
        """
        if not isinstance(camera, MjvCamera):
            camera_id = camera
            if isinstance(camera_id, str):
                camera_id = mj_name2id(self.model, mjtObj.mjOBJ_CAMERA.value, camera_id)
                if camera_id == -1:
                    raise ValueError(f'The camera "{camera}" does not exist.')
            if camera_id < -1 or camera_id >= self.model.ncam:
                raise ValueError(
                    f"The camera id {camera_id} is out of range [-1, {self.model.ncam})."
                )

            # Render camera.
            camera = MjvCamera()
            camera.fixedcamid = camera_id

            # Defaults to mjCAMERA_FREE, otherwise mjCAMERA_FIXED refers to a
            # camera explicitly defined in the model_bindings.
            if camera_id == -1:
                camera.type = mjtCamera.mjCAMERA_FREE
                mjv_defaultFreeCamera(self.model, camera)
            else:
                camera.type = mjtCamera.mjCAMERA_FIXED

        scene_option = scene_option or self._scene_option
        mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            camera,
            mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )

    def close(self) -> None:
        """Frees the resources used by the renderer.

        This method can be used directly:

        ```python
        renderer = Renderer(...)
        # Use renderer.
        renderer.close()
        ```

        or via a context manager:

        ```python
        with Renderer(...) as renderer:
          # Use renderer.
        ```
        """
        if hasattr(self, "_gl_context") and self._gl_context:
            self._gl_context.free()
        self._gl_context = None
        if hasattr(self, "_mjr_context") and self._mjr_context:
            self._mjr_context.free()
        self._mjr_context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback  # Unused.
        self.close()

    def __del__(self) -> None:
        self.close()


class MjBatchRenderer:
    """
    Reference:
    https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjbatchrenderer.pyx
    https://github.com/openai/mujoco-py/commit/aa82c0e555d28813394f04ee8a8c2fc6b18d6b3f
    https://github.com/openai/mujoco-py/pull/94
    https://github.com/openai/mujoco-py/pull/246/files#diff-2e59ed7fefb358a2579ea4033f71379d3659858b63dcb4a0d161fca9b3e43522
    """

    pass


class MultithreadOpenGLRenderer(MultithreadRenderer):
    def __init__(
        self,
        env,  #: "MuJoCoVectorEnv",
        renderer_cls: type[MjAbstractRenderer] = MjOpenGLRenderer,
        max_render_contexts: int | None = None,
        namespace: str = "robot_0/",
        width: int = 1280,
        height: int = 720,
        **kwargs: Any,
    ) -> None:
        self.width = width
        self.height = height
        self.namespace = namespace

        self.render_outputs: list[np.ndarray] | None = None

        super().__init__(**prepare_locals_for_super(locals()))

    @staticmethod
    def process_request(
        renderer: MjOpenGLRenderer, request: Any, output_queue: Queue, **process_request_kwargs
    ) -> None:
        idx, camera, data, mode = request

        if mode == "rgb":
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        elif mode == "depth":
            renderer.enable_depth_rendering()
        elif mode == "segmentation":
            renderer.enable_segmentation_rendering()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        renderer.update(data, camera=camera)
        img = renderer.render(**process_request_kwargs)
        output_queue.put((idx, img))

    def render(
        self,
        camera: str = "camera_rgb",
        mode: Literal["rgb", "depth", "segmentation"] = "rgb",
        add_namespace: bool = True,
    ):
        for idx, (model, data) in enumerate(zip(self.env.mj_models, self.env.mj_datas)):
            self.model_id_to_render_input_queue[id(model)].put(
                (idx, (self.namespace if add_namespace else "") + camera, data, mode)
            )

        idx_render_tuples = [
            self.render_output_queue.get(block=True) for _ in range(len(self.env.mj_models))
        ]

        idx_render_tuples.sort(key=lambda x: x[0])
        self.render_outputs = [img for _, img in idx_render_tuples]

        return self.render_outputs

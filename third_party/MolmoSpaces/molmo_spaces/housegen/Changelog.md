## Fixes

### 11-21-25

- Fixed walls' visuals being pure-only geoms. It was working ok for the colliders, but had to group
  the visual in the same body as the colliders, not the worldbody.

Previously it was like this:

```xml
  <worldbody>
    <geom name="floor" class="__STRUCTURAL_MJT__" size="0 0 0.01" type="plane" solimp="0.9 0.95" />
    <geom name="wall_4_8" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="PureWhite" mesh="wall_4_8" />
    <geom name="wall_5_10" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="PureWhite" mesh="wall_5_10" />
    <geom name="wall_4_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="PureWhite" mesh="wall_4_0" />
    <geom name="wall_7_19" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="PureWhite" mesh="wall_7_19" />
    <geom name="wall_7_21" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="PureWhite" mesh="wall_7_21" />
    <geom name="wall_0_23" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="0" material="RoseGraniteTiles" mesh="wall_0_23" />
    <light pos="1 -1 1.5" dir="-0.57735 0.57735 -0.57735" directional="true" diffuse="0.5 0.5 0.5" />
    <body name="room_4">
      <geom name="room_4_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="WornWood" mesh="room_4" />
    </body>
    <body name="room_5">
      <geom name="room_5_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="OrangeWood" mesh="room_5" />
    </body>
    <body name="wall_4_0">
      <geom name="wall_4_0_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="PureWhite" mesh="wall_4_0" />
      <geom name="wall_4_0_collision_0" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_0" />
      <geom name="wall_4_0_collision_1" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_1" />
      <geom name="wall_4_0_collision_2" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_2" />
    </body>
    ...
  </worldbody>
```

Now it should be like this:

```xml
  <worldbody>
    <geom name="floor" class="__STRUCTURAL_MJT__" size="0 0 0.01" type="plane" solimp="0.9 0.95" />
    <light pos="1 -1 1.5" dir="-0.57735 0.57735 -0.57735" type="directional" diffuse="0.5 0.5 0.5" />
    <body name="room_4">
      <geom name="room_4_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="WornWood" mesh="room_4" />
    </body>
    <body name="room_5">
      <geom name="room_5_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="OrangeWood" mesh="room_5" />
    </body>
    <body name="wall_4_0">
      <geom name="wall_4_0_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="PureWhite" mesh="wall_4_0" />
      <geom name="wall_4_0_collision_0" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_0" />
      <geom name="wall_4_0_collision_1" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_1" />
      <geom name="wall_4_0_collision_2" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_0_collision_mesh_2" />
    </body>
    <body name="wall_4_1">
      <geom name="wall_4_1_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="PureWhite" mesh="wall_4_1" />
      <geom name="wall_4_1_collision_0" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_1" />
    </body>
    <body name="wall_4_2">
      <geom name="wall_4_2_visual_0" class="__VISUAL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" mass="1e-08" material="PureWhite" mesh="wall_4_2" />
      <geom name="wall_4_2_collision_0" class="__STRUCTURAL_WALL_MJT__" pos="0 0 0" quat="0.707107 0.707107 0 0" type="mesh" solimp="0.9 0.95" mesh="wall_4_2" />
    </body>
    ...
  </worldbody>
```

### 11-20-25

- Added an extra offset to some object categories only for holodeck houses, as it seems that the
  location used for the house json was computed using a different version of the assets, giving a
  different bounding box, which might caused the objects from these categories to have a wrong
  position in the house json. This issue is being tracked [here](https://github.com/allenai/mujoco-thor/issues/314)

### 11-19-25

- Added a processing step during exporting to delete the cloned versions of some assets, as it seems
  some houses have duplicated of some objects that spawn in the same position, which makes the
  simulation slow down to around 1% realtime factor, due to too many contacts being handled for only
  these problematic objects. This issue is being tracked [here](https://github.com/allenai/mujoco-thor/issues/302)


### 11-17-25

- Fixed an issue with a pair of assets that have wrong positions in procthor houses. The asset types
are `Toilet_2` and `Desk_306_1`, which look fine in iTHOR houses, but have bad positions when used
in ProcTHOR-10k houses. The fix for now was to add an offset to the model when adding it to the
house. This was introduced in `401b5726cd3d7c28761fdf26ab2f86c651b48e5f` for Procthor-10k, Procthor-Objaverse
and Holodeck-Objaverse houses only, as this works fine for iTHOR houses (the positions come good
from the Unity scene). This issue is being tracked [here](https://github.com/allenai/mujoco-thor/issues/318)

### 11-14-25

- Tuned the `solimp` params for both `pen` and `pencil` assets categories, as well as changed the
  primitive collider type from capsule to cylinder to make it more stable. This was changed in commit
  `1e32f1ab32002c1c447876d8f3f284c234551fd0`. These changes fix the issue we had with pens and pencils
  going through some surfaces from tables and desks when doing the settle phase when exporting the houses.
  This issue is being tracked [here](https://github.com/allenai/mujoco-thor/issues/319)

### 11-10-25

- Fixed an issue with z-fighting on some walls for ProcTHOR-10k scenes. The fix was added in commit
  `ee8f309f4cf0652ff5a77034e99485b2fedd37f0`, and the issue is being tracked in [this](https://github.com/allenai/mujoco-thor/issues/315)
  github issue.

### 11-07-25

- Fixed an issue with the walls not using the correct colliders when adding holes to the doors. This
  was fixed in commit `4e8ab58fafec39f6ac6f36e6c2e47e847d4d868d`, and the issue is being tracked in
  [this](https://github.com/allenai/mujoco-thor/issues/317) github issue.

### 11-05-25

- Updated default joint damping for free joints for all bodies in the scenes from `0.1` to `0.001`,
  which helped make motion a bit more realistic.
- Added `margin=0.001` to all THOR objects that use mesh colliders. This fixes stability issues
  and avoids jitter (only applied to mesh geoms bc for primitives would have caused stability
  issues, and those were already stable and not jittery)


## Gotchas to keep in mind with the house generator code

- We had issues with MjSpec when using `mujoco=3.3.2`, when handling defaults and using it to set
  positions after settling the simulation for a bit. This might be fixed in the latest version, but
  for now we added some patches:

    1. We set the defaults after the objects is saved with MjSpec. So first comes `spec.to_xml()`,
       and then comes a call to `_apply_patches_to_mjcf_house`, which effectively removes the
       extra dummy|empty defaults that MjSpec didn't remove for us. This made the loading of the
       model crash, so we have to patch it after saving the model with mjspec.
    2. We had to set the margin property to fix an issue with jittering. We just had to modify the
       default for the `__DYNAMIC_MJT__` class, however this fails with MjSpec. That's why we also
       had to manually set the default margin in the corresponding class default, and remove the
       extra `margin` attribute that mjspec was adding to the geoms (it wasn't following the
       defaults, but instead copying the parameters to the geoms T_T). This is made in the
       `_apply_defaults_to_house_mjcf` method, and for now the only attribute in the geoms that is
       being removed is the margin, which MjSpec adds it even though it was defined in its default
       for that classname.

import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

import molmo_spaces.utils.distance_transform_utils as dtutils
from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.planner.abstract import Planner
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

log = logging.getLogger(__name__)


class AStarPlannerConfig(Config):
    # A* planner parameters
    downscale_factor: int = 5
    blacklist_blob_width_m: float = 0.35
    max_start_goal_distance: int = 40  # Search radius in grid cells (~1m with default settings)

    # Map computation
    agent_radius: float = 0.5
    px_per_m: int = 200


class AStarPlanner(Planner):
    def __init__(
        self,
        config: AStarPlannerConfig,
        model_path: Path | str,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_path = model_path

        self.downscale = self.config.downscale_factor

        self._map: ProcTHORMap | None = None
        self._grid_spacing: float | None = None
        self._downscaled_grid: np.ndarray[bool] | None = None
        self._dt: np.ndarray[float] | None = None
        self._graph: nx.Graph | None = None

        self.blacklist = []
        self.blacklist_blob_width_m = self.config.blacklist_blob_width_m
        self._blacklist_blob_width = None

    @property
    def map(self):
        if self._map is None:
            self._grid_spacing = None
            self._downscaled_grid = None
            self._dt = None
            self._graph = None

            if "ithor" in self.model_path:
                self._map = iTHORMap.from_mj_model_path(
                    model_path=self.model_path,
                    agent_radius=self.config.agent_radius,
                    px_per_m=self.config.px_per_m,
                )
            elif "procthor" in self.model_path or "holodeck" in self.model_path:
                self._map = ProcTHORMap.from_mj_model_path(
                    model_path=self.model_path,
                    px_per_m=self.config.px_per_m,
                    agent_radius=self.config.agent_radius,
                )
            else:
                raise ValueError(f"Unknown scene type: {self.model_path}")

        return self._map

    @property
    def grid_spacing(self):
        if self._grid_spacing is None:
            self._grid_spacing = self.downscale / self.map.px_per_m
        return self._grid_spacing

    @property
    def blacklist_blob_width(self):
        if self._blacklist_blob_width is None:
            base_black_list_blob_width = int(round(self.blacklist_blob_width_m / self.grid_spacing))
            self._blacklist_blob_width = base_black_list_blob_width + (
                1 - base_black_list_blob_width % 2
            )
        return self._blacklist_blob_width

    def discretize_location(self, location):
        return np.floor(self.map.pos_m_to_px(location) / self.downscale).astype(np.int32)

    def apply_black_list(self) -> None:
        self._dt = None  # force recompute, now or in next plan

        locs = Counter(tuple(self.discretize_location(loc)) for loc in self.blacklist)
        if len(locs) == 0:
            return

        width = self.blacklist_blob_width
        hsize = (width - 1) // 2
        for loc in locs:
            raw_rmin = loc[0] - hsize
            rmin = max(raw_rmin, 0)
            rmax = min(raw_rmin + width, self.dt.shape[0] - 1)

            raw_cmin = loc[1] - hsize
            cmin = max(raw_cmin, 0)
            cmax = min(raw_cmin + width, self.dt.shape[1] - 1)

            # Make it lower with overlapping contributions, why not?
            self.dt[rmin:rmax, cmin:cmax] *= np.power(0.5, locs[loc])

        self._graph = None  # force recompute in next plan

    @property
    def downscaled_grid(self):
        if self._downscaled_grid is None:
            self._downscaled_grid = self._make_downscaled_grid()

        return self._downscaled_grid

    @property
    def dt(self):
        if self._dt is None:
            self._dt = dtutils.make_distance_transform(self.downscaled_grid, self.grid_spacing)

        return self._dt

    @property
    def graph(self):
        if self._graph is None:
            self._graph = dtutils.make_grid_graph(self.downscaled_grid, self.dt, weight_exp=2)

        return self._graph

    def _make_downscaled_grid(self):
        img = self.map.occupancy
        if len(img.shape) == 3:
            img = img[..., 0]

        # Negate to convert into a navigable space grid
        # grid = ~(img.copy().astype(bool))
        # Actually not, because occupancy==True means free space
        grid = img.copy().astype(bool)

        # Make padding non-navigable (set to False)
        padded = np.zeros(
            (
                grid.shape[0] + (self.downscale - grid.shape[0] % self.downscale),
                grid.shape[1] + (self.downscale - grid.shape[1] % self.downscale),
            ),
            dtype=bool,
        )
        padded[: grid.shape[0], : grid.shape[1]] = grid

        # If any pixel in high-res patch non-navigable, output pixel also non-navigable
        return (
            padded.reshape(
                padded.shape[0] // self.downscale,
                self.downscale,
                padded.shape[1] // self.downscale,
                self.downscale,
            )
            .min(axis=1)
            .min(axis=-1)
        )

    def get_discrete_location(self, pos):
        discrete_pos = self.discretize_location(pos)

        def find_close(missing):
            # TODO check the close location is actually in the same room (e.g. using room polygon)
            for search_range in range(1, self.config.max_start_goal_distance + 1):
                for shiftr in range(-search_range, search_range + 1):
                    for shiftc in range(-search_range, search_range + 1):
                        if shiftr != search_range and shiftc != search_range:
                            continue
                        cur_goal = (missing[0] + shiftr, missing[1] + shiftc)
                        if cur_goal in self.graph:
                            return cur_goal
            return None

        if tuple(discrete_pos) not in self.graph:
            discrete_pos = find_close(discrete_pos)

        return discrete_pos

    def _compute_plan(self, start, goal):
        try:
            waypoints, _, _ = dtutils.make_discrete_path(
                self.graph,
                start[0],
                start[1],
                goal[0],
                goal[1],
                self.dt,
                3,
                self.grid_spacing,
                0.6,
            )

            pixel_waypoints = np.array(waypoints) * self.downscale
            world_waypoints = self.map.pos_px_to_m(pixel_waypoints)[:, :2]

            return world_waypoints
        except nx.NetworkXUnfeasible as e:
            log.error(f"Error in compute_plan: {e}")
            return None

    def motion_plan(
        self,
        target_pos: np.ndarray,
        robot_view: RobotView,
        **kwargs,
    ):
        init_pos = robot_view.base.pose[:3, 3]

        discrete_init = self.get_discrete_location(init_pos)
        if discrete_init is None:
            raise ValueError("Non-plannable starting position")

        discrete_goal = self.get_discrete_location(target_pos)
        if discrete_goal is None:
            raise ValueError("Non-plannable target position")

        return self._compute_plan(discrete_init, discrete_goal)

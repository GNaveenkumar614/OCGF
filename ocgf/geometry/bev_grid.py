"""The shared BEV grid.

BEV frame = LiDAR frame: x forward, y left, z up, ego at the LiDAR origin.
Default grid: x in [0, 50) m, y in [-20, 20) m, 0.25 m cells -> arrays of shape [200, 160], indexed [ix, iy].
Example: (12.4, -1.5) m -> (49, 74).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEVGrid:
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -20.0
    y_max: float = 20.0
    res: float = 0.25

    @property
    def shape(self) -> tuple[int, int]:
        """(nx, ny): cells along x (forward) and along y (left)."""
        return (int(round((self.x_max - self.x_min) / self.res)),
                int(round((self.y_max - self.y_min) / self.res)))

    def index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices ix = floor((x - x_min) / res), iy = floor((y - y_min) / res), without bounds checks."""
        ix = np.floor((np.asarray(x, dtype=np.float64) - self.x_min) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y, dtype=np.float64) - self.y_min) / self.res).astype(np.int64)
        return ix, iy

    def inside(self, x, y) -> np.ndarray:
        """True where (x, y) falls in a grid cell. Uses the same floor as index(), so the two always agree."""
        ix, iy = self.index(x, y)
        nx, ny = self.shape
        return (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates: xs [nx] and ys [ny]."""
        nx, ny = self.shape
        return (self.x_min + (np.arange(nx) + 0.5) * self.res,
                self.y_min + (np.arange(ny) + 0.5) * self.res)

    def center_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates as two [nx, ny] arrays."""
        xs, ys = self.centers()
        return np.meshgrid(xs, ys, indexing="ij")

    @staticmethod
    def to_display(bev: np.ndarray) -> np.ndarray:
        """Flip the last two axes for display: forward points up, left is on the left. Use only for display."""
        return np.flip(bev, axis=(-2, -1))

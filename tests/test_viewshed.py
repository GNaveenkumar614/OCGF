"""Viewshed: shadows behind a synthetic box (a required test), a curb's short shadow, nothing hidden in front."""
import numpy as np

from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.raster import box_cells
from ocgf.geometry.viewshed import viewshed

GRID = BEVGrid()
GROUND = -1.73                                   # LiDAR about 1.73 m above flat ground


def flat():
    return np.full(GRID.shape, GROUND)


def with_box(box, top):
    occ = np.full(GRID.shape, np.nan)
    ix, iy = box_cells(box, GRID)
    occ[ix, iy] = top
    return occ


def cell(x, y):
    ix, iy = GRID.index(x, y)
    return int(ix), int(iy)


def test_open_flat_ground_is_all_visible():
    assert viewshed(GRID, (0, 0, 0), flat(), np.full(GRID.shape, np.nan)).all()


def test_a_car_hides_the_ground_behind_it_beyond_the_grid():
    # a 1.5 m tall car at 10 m: its top is 0.23 m below the sensor, so the shadow reaches about 75 m
    box = (10.0, 0.0, GROUND + 0.75, 4.0, 1.8, 1.5, 0.0)
    vis = viewshed(GRID, (0, 0, 0), flat(), with_box(box, GROUND + 1.5))
    for x in (13.0, 20.0, 35.0, 49.0):
        assert not vis[cell(x, 0.0)], f"ground behind the car at x={x} should be hidden"
    for x in (3.0, 6.0, 7.5):
        assert vis[cell(x, 0.0)], f"ground in front of the car at x={x} should be visible"
    assert vis[cell(20.0, 6.0)] and vis[cell(10.0, 4.0)]                 # beside the car's wedge
    # the wedge widens with distance: the car's near edge spans +-6.4 deg (0.9 m at 8 m), about +-5 m at 45 m,
    # plus up to half a cell (rays crossing any part of an occluder cell are blocked)
    assert not vis[cell(45.0, 3.0)] and not vis[cell(45.0, 4.5)] and vis[cell(45.0, 7.5)]


def test_a_curb_hides_about_a_metre():
    curb = (8.0, 0.0, GROUND + 0.075, 0.25, 6.0, 0.15, 0.0)
    vis = viewshed(GRID, (0, 0, 0), flat(), with_box(curb, GROUND + 0.15))
    assert not vis[cell(8.5, 0.0)]                                        # just behind the curb
    assert vis[cell(9.5, 0.0)] and vis[cell(20.0, 0.0)]                   # about 0.8 m of shadow, then visible
    assert vis[cell(7.6, 0.0)]


def test_unknown_ground_is_never_visible_and_the_origin_matters():
    ground = flat()
    ground[cell(20.0, 0.0)] = np.nan
    vis = viewshed(GRID, (0, 0, 0), ground, np.full(GRID.shape, np.nan))
    assert not vis[cell(20.0, 0.0)]
    box = (10.0, 0.0, GROUND + 0.75, 4.0, 1.8, 1.5, 0.0)
    from_left = viewshed(GRID, (0, 8.0, 0), flat(), with_box(box, GROUND + 1.5))
    assert from_left[cell(20.0, 0.0)] or not from_left[cell(20.0, -8.0)]  # the shadow moves with the sensor
    assert not from_left[cell(20.0, -8.0)]

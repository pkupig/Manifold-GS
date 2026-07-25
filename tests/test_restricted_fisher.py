import numpy as np
from manifold_gs.restricted_fisher import classify_patch, finite_difference_fisher, patch_normals, scene_threshold


def test_fisher_center_difference_and_mask():
    minus = np.zeros((2, 2, 3)); plus = np.zeros((2, 2, 3)); plus[..., 0] = 0.2
    fisher, count = finite_difference_fisher(plus, minus, .1, np.array([[True, False], [True, False]]))
    assert count == 2 and np.isclose(fisher, 1.0)


def test_fisher_unresolved_and_threshold():
    assert classify_patch(1., 2, 1000, .5) == 'insufficient_views'
    assert classify_patch(float('nan'), 3, 1000, .5) == 'numerically_unresolved'
    assert classify_patch(.1, 3, 256, .2) == 'weakly_identified'
    assert classify_patch(.2, 3, 256, .2) == 'fisher_supported'
    assert np.isclose(scene_threshold([{'fisher': 1., 'views': 3, 'pixels': 256}, {'fisher': 11., 'views': 3, 'pixels': 256}, {'fisher': 99., 'views': 1, 'pixels': 999}]), 2.0)


def test_patch_normals_sign_independent_plane():
    xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,1],[3,0,1],[2,0,2]], float)
    normals = patch_normals(xyz, np.array([7,7,7,8,8,8]))
    assert np.isclose(abs(normals[7][2]), 1.0)
    assert np.isclose(abs(normals[8][1]), 1.0)


def test_scene_threshold_ignores_serialised_unresolved_record():
    records = [
        {"fisher": None, "views": 3, "pixels": 256},
        {"fisher": 2.0, "views": 3, "pixels": 256},
    ]
    assert np.isclose(scene_threshold(records), 2.0)

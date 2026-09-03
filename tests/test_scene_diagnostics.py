import numpy as np

from general_motion_retargeting.scene_diagnostics import alignment_sanity_check, summarize_scene_diagnostics


def test_alignment_sanity_accepts_same_geometry():
    points = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    result = alignment_sanity_check(points, points.copy(), points.copy())
    assert result["passed"]
    assert result["max_bound_error"] == 0.0


def test_alignment_sanity_rejects_offset_and_scale():
    visual = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    interaction = visual + np.array([0.2, 0, 0])
    collision = visual * 0.5
    result = alignment_sanity_check(visual, interaction, collision)
    assert not result["passed"]
    assert result["max_bound_error"] > 0.1


def test_summary_reports_contacts_and_collisions():
    records = [
        {"active_scene_collision_pairs": 2, "minimum_scene_distance": 0.01, "maximum_penetration": 0.0,
         "scene_collision_query_runtime_seconds": 0.002, "qp_solve_runtime_seconds": 0.01, "qp_failures": []},
        {"active_scene_collision_pairs": 4, "minimum_scene_distance": -0.002, "maximum_penetration": 0.002,
         "scene_collision_query_runtime_seconds": 0.003, "qp_solve_runtime_seconds": 0.02, "qp_failures": ["failed"]},
    ]
    contacts = [{"contacts": {"left_butt": {"state": "STATIC", "signed_distance": 0.004, "surface_id": "seat"}}},
                {"contacts": {"left_butt": {"state": "STATIC", "signed_distance": 0.006, "surface_id": "seat"}}}]
    summary = summarize_scene_diagnostics(records, contacts)
    assert summary["qp_failure_count"] == 1
    assert summary["max_active_scene_collision_pairs"] == 4
    assert summary["left_butt_contact_ratio"] == 1.0
    assert summary["contact_channels"]["left_butt"]["surface_ids"] == ["seat"]

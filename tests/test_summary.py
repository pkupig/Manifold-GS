from scripts.summarize_experiment_manifest import compute_checks


def test_compute_checks_passes_consistent_paired_improvements() -> None:
    comparison = {
        "baseline": "vanilla",
        "matched_geometry_baseline": "tangent",
        "candidate": "manifold_full",
        "minimum_relative_improvement": {
            "chamfer_l1": 0.15,
            "normal_accuracy_median_deg": 0.20,
            "normalized_kernel_varifold": 0.20,
        },
        "asset_minimum_relative_improvement": {
            "mesh_chamfer_l1": 0.10,
            "mesh_normal_accuracy_median_deg": 0.10,
        },
        "guardrails": {
            "maximum_mean_psnr_drop_db": 0.3,
            "maximum_mean_ssim_drop": 0.01,
        },
        "incremental_vs_matched_baseline": {
            "minimum_symmetry_improvement": 0.10,
            "maximum_primary_metric_regression": 0.02,
            "maximum_chart_accepted_mass_drop": 0.02,
        },
    }
    rows = []
    for seed in range(3):
        rows.append({
            "scene": "test", "seed": seed,
            "vanilla": {
                "chamfer_l1": 1.0, "normal_accuracy_median_deg": 1.0,
                "normalized_kernel_varifold": 1.0, "mesh_chamfer_l1": 1.0,
                "mesh_normal_accuracy_median_deg": 1.0, "psnr": 20.0,
                "ssim": 0.8, "symmetry": 1.0, "chart_accepted_mass_fraction": 0.5,
            },
            "tangent": {
                "chamfer_l1": 0.9, "normal_accuracy_median_deg": 0.9,
                "normalized_kernel_varifold": 0.9, "mesh_chamfer_l1": 0.9,
                "mesh_normal_accuracy_median_deg": 0.9, "psnr": 20.0,
                "ssim": 0.8, "symmetry": 0.9, "chart_accepted_mass_fraction": 0.6,
            },
            "manifold_full": {
                "chamfer_l1": 0.7, "normal_accuracy_median_deg": 0.7,
                "normalized_kernel_varifold": 0.7, "mesh_chamfer_l1": 0.7,
                "mesh_normal_accuracy_median_deg": 0.7, "psnr": 19.9,
                "ssim": 0.795, "symmetry": 0.7, "chart_accepted_mass_fraction": 0.65,
            },
        })

    checks, status = compute_checks(rows, comparison)

    assert status == "PASS"
    assert {check["decision"] for check in checks.values()} == {"PASS"}

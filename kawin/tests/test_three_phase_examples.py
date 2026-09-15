import numpy as np
import pytest

from examples.ternaryExamples import IllingworthTernaryThreePhaseStable as three_phase_example


def _baseline_config():
    """Returns mutable-safe values that must survive one temporary run override."""
    return {
        "case_name": three_phase_example.CASE_NAME,
        "liquid_diffusivity": three_phase_example.FECRNI_LIQUID_DIFFUSIVITY_MATRIX.copy(),
        "eta_samples": three_phase_example.ETA_SAMPLES.copy(),
        "phase_molar_volumes": three_phase_example.PHASE_MOLAR_VOLUMES,
    }


def _assert_baseline_config(baseline):
    """Checks that ``run_case`` did not leak temporary configuration changes."""
    assert three_phase_example.CASE_NAME == baseline["case_name"]
    assert np.array_equal(three_phase_example.FECRNI_LIQUID_DIFFUSIVITY_MATRIX, baseline["liquid_diffusivity"])
    assert np.array_equal(three_phase_example.ETA_SAMPLES, baseline["eta_samples"])
    assert three_phase_example.PHASE_MOLAR_VOLUMES == baseline["phase_molar_volumes"]


def test_stable_three_phase_fixture_has_frozen_fecrni_baseline():
    assert three_phase_example.CASE_NAME == three_phase_example.CASE_FE_CR_NI_PYCALPHAD
    assert three_phase_example.PHASES_FOR_MODEL == ("FCC_A1", "LIQUID", "BCC_A2")


def test_fecrni_pycalphad_case_requires_explicit_liquid_diffusivity():
    baseline = _baseline_config()
    with pytest.raises(ValueError, match="FECRNI_LIQUID_DIFFUSIVITY_MATRIX"):
        three_phase_example.run_case(
            {
                "case_name": "fe_cr_ni_pycalphad",
                "fecrni_liquid_diffusivity_matrix": None,
                "run_preflight": False,
                "run_solve": False,
                "eta_samples": np.linspace(0.0, 1.0, 3),
            },
            make_plots=False,
        )

    _assert_baseline_config(baseline)


def test_fecrni_pycalphad_case_builds_with_fixed_liquid_diffusivity():
    baseline = _baseline_config()
    liquid_matrix = np.asarray([[1.0e-9, 0.0], [0.0, 1.0e-9]], dtype=np.float64)

    result = three_phase_example.run_case(
        {
            "case_name": "fe_cr_ni_pycalphad",
            "fecrni_liquid_diffusivity_matrix": liquid_matrix,
            "run_preflight": False,
            "run_solve": False,
            "eta_samples": np.linspace(0.0, 1.0, 3),
            "diffusivity_interpolation": "nearest",
            "bulk_diffusivity_points": np.empty((0, 2), dtype=np.float64),
            "nodes": 31,
            "phase_nodes": (5, 5, 5),
            "phase_molar_volumes": (7.1e-6, 7.8e-6, 7.3e-6),
            "verbose": False,
        },
        make_plots=False,
    )

    model = result["model"]
    liquid_diffusivity = result["bulk_thermodynamics"].getInterdiffusivity(
        np.asarray([[0.43, 0.37], [0.44, 0.35]], dtype=np.float64),
        1650.0,
        phase="LIQUID",
    )

    assert model.phases == ("FCC_A1", "LIQUID", "BCC_A2")
    assert np.array_equal(model._phaseMolarVolumes, [7.1e-6, 7.8e-6, 7.3e-6])
    assert model._hasExplicitPhaseMolarVolumes
    assert np.allclose(model.getInterfacePositions(), [50.0e-6, 60.0e-6])
    assert result["surrogate_ab"].tieline_phases == ("FCC_A1", "LIQUID")
    assert result["surrogate_bc"].tieline_phases == ("LIQUID", "BCC_A2")
    assert liquid_diffusivity.shape == (2, 2, 2)
    assert np.allclose(liquid_diffusivity, np.broadcast_to(liquid_matrix, (2, 2, 2)))
    _assert_baseline_config(baseline)

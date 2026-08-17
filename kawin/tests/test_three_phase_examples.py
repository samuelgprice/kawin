import numpy as np
import pytest

from examples.ternaryExamples import IllingworthTernaryThreePhaseNiTiNb_TC as three_phase_example


def test_three_phase_example_defaults_to_ni_ti_nb_tc_case():
    assert three_phase_example.CASE_NAME == three_phase_example.CASE_NI_TI_NB_TC
    assert three_phase_example.PHASES_FOR_MODEL == ("BCC_B2", "LIQUID", "BCC_B2")


def test_fecrni_pycalphad_case_requires_explicit_liquid_diffusivity():
    with pytest.raises(ValueError, match="FECRNI_LIQUID_DIFFUSIVITY_MATRIX"):
        three_phase_example.run_case(
            {
                "case_name": "fe_cr_ni_pycalphad",
                "run_preflight": False,
                "run_solve": False,
                "eta_samples": np.linspace(0.0, 1.0, 3),
            },
            make_plots=False,
        )

    assert three_phase_example.CASE_NAME == three_phase_example.CASE_NI_TI_NB_TC


def test_fecrni_pycalphad_case_builds_with_fixed_liquid_diffusivity():
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
    assert np.allclose(model.getInterfacePositions(), [50.0e-6, 60.0e-6])
    assert result["surrogate_ab"].tieline_phases == ("FCC_A1", "LIQUID")
    assert result["surrogate_bc"].tieline_phases == ("LIQUID", "BCC_A2")
    assert liquid_diffusivity.shape == (2, 2, 2)
    assert np.allclose(liquid_diffusivity, np.broadcast_to(liquid_matrix, (2, 2, 2)))
    assert three_phase_example.CASE_NAME == three_phase_example.CASE_NI_TI_NB_TC

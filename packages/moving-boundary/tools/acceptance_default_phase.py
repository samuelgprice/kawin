"""Real default-phase endpoint/equilibrium sampling, including a single-phase probe."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import numpy as np


def sample_default_phase(inputs, configuration, *, generation_guard):
    """Sample the frozen three probes using the actual full-phase facade.

    Reference code is imported unchanged from detached 06dc7b4, so the invalid
    probe executes its unused examples.debugInPlace import. Package generation
    uses the installed adapter under the same import/read guard as O05. No
    surrogate, cached endpoint or fake thermodynamic provider substitutes for
    equilibrium. Ordered labels and invalid -1 endpoints are part of the gate.
    """
    assert configuration["use_default_phases"] is True
    package = inputs.role == "package"
    with generation_guard() if package else nullcontext({}) as audit:
        if package:
            from moving_boundary.providers import pycalphad as source
            import moving_boundary
            assert Path(source.__file__).resolve().is_relative_to(Path(moving_boundary.__file__).resolve().parent)
        else:
            path = inputs.root / "examples/ternaryExamples/pycalphad_default_phase_adapter.py"
            spec = importlib.util.spec_from_file_location("frozen_default_phase_adapter",path)
            source = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(source)
            assert Path(source.__file__).resolve() == path.resolve()
        facade = source.create_pycalphad_thermodynamics_source(
            Path(inputs.artifacts) / configuration["database_input"],configuration["elements"],
            configuration["interface_phases"],use_default_phases=True,g_offset=configuration["g_offset"])
        assert type(facade).__name__ == "PycalphadDefaultPhaseThermodynamics"
        universe = tuple(facade.equilibrium_phases)
        assert universe == tuple(configuration["equilibrium_phases"])
        expected = tuple(configuration["interface_phases"])
        probes = configuration["probes"]
        temperature = configuration["temperature"]
        alpha, beta, valid, labels, amounts, compositions, potentials, indices = [],[],[],[],[],[],[],[]
        for probe in probes:
            left,right,metadata = facade.getInterfacialComposition(probe,temperature,returnMeta=True)
            equilibrium = facade.getEquilibriumData(probe,temperature)
            observed = tuple(metadata["endpoint_phases"])
            accepted = observed == expected and len(metadata["endpoints"]) == 2
            if not accepted:
                assert np.array_equal(left,-np.ones(len(configuration["elements"])-1))
                assert np.array_equal(right,-np.ones(len(configuration["elements"])-1))
            alpha.append(left)
            beta.append(right)
            valid.append(accepted)
            labels.append(list(observed))
            amounts.append([equilibrium["phase_amounts"].get(phase,0.) for phase in universe])
            compositions.append([equilibrium["phase_compositions"].get(phase,np.full(len(configuration["elements"]),np.nan))
                                 for phase in universe])
            potentials.append(equilibrium["chemical_potentials"])
            indices.append([universe.index(phase) for phase in observed]+[-1]*(2-len(observed)))
        assert valid == configuration["expected_valid"], (valid,labels)
        assert any(valid) and not all(valid), "Both valid tie lines and an invalid probe are required"
        if not package:
            assert "examples.debugInPlace" in sys.modules, "Frozen invalid branch was not exercised"
        arrays = {
            "provider/interfaceData/alpha":np.asarray(alpha),
            "provider/interfaceData/beta":np.asarray(beta),
            "provider/valid":np.asarray(valid,dtype=bool),
            "provider/endpoint_phase_indices":np.asarray(indices,dtype=np.int64),
            "provider/phase_amounts":np.asarray(amounts),
            "provider/full_compositions":np.asarray(compositions),
            "provider/chemical_potentials":np.asarray(potentials),
        }
        contract = dict(facade_class=type(facade).__name__,elements=configuration["elements"],
            interface_phases=list(expected),equilibrium_phases=list(universe),
            endpoint_phases=labels,valid_probes=sum(valid),invalid_probes=len(valid)-sum(valid),
            use_default_phases=True)
        audit.update(adapter_module=source.__name__,adapter_origin=source.__file__)
    return arrays,contract,audit

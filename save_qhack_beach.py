import json
import pennylane as qml
import pennylane.numpy as np
import time


def potential_energy_surface(symbols, bond_lengths):
    """Calculates the molecular energy over various bond lengths (AKA the
    potential energy surface) using the Hartree Fock method.

    Args:
        symbols (list(string)):
            A list of atomic symbols that comprise the diatomic molecule of interest.
        bond_lengths (numpy.tensor): Bond lengths to calculate the energy over.


    Returns:
        hf_energies (numpy.tensor):
            The Hartree Fock energies at every bond length value.
    """

    geometry_list = [np.array([[0, 0, 0], [0, 0, bond_length]], requires_grad=False) for bond_length in bond_lengths]
    molecules = [qml.qchem.Molecule(symbols, geometry) for geometry in geometry_list]
    hf_energies = [qml.qchem.hf_energy(molecule)(molecule.alpha) for molecule in molecules]
    return np.array(hf_energies)


def ground_energy(hf_energies):
    """Finds the minimum energy of a molecule given its potential energy surface.

    Args:
        hf_energies (numpy.tensor):

    Returns:
        (float): The minumum energy in units of hartrees.
    """

    ind = np.argmin(hf_energies)
    return hf_energies[ind]


def reaction():
    """Calculates the energy of the reactants, the activation energy, and the energy of
    the products in that order.

    Returns:
        (numpy.tensor): [E_reactants, E_activation, E_products]
    """
    molecules = {
        "H2":
            {"symbols": ["H", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(0.5, 9.3, 0.3)},
        "Li2":
            {"symbols": ["Li", "Li"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(3.5, 8.3, 0.3)},
        "LiH":
            {"symbols": ["Li", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(2.0, 6.6, 0.3)}
    }

    for molecule in molecules.keys():
        mol_data = molecules[molecule]
        surface = potential_energy_surface(mol_data['symbols'], mol_data['bond lengths'])
        mol_data['E0'] = ground_energy(surface)
        mol_data['E_dissociation'] = abs(mol_data['E0'] - surface[-1])

    E_reactants = molecules['H2']['E0'] + molecules['Li2']['E0']
    E_activation = E_reactants + molecules['H2']['E_dissociation'] + molecules['Li2']['E_dissociation']
    E_products = 2 * molecules['LiH']['E0']

    return np.array([E_reactants, E_activation, E_products])


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    output = reaction().tolist()
    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.allclose(solution_output, expected_output, rtol=1e-3)


res = reaction()
print(res)

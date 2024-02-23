import json
import pennylane as qml
import pennylane.numpy as np


dev = qml.device('default.qubit', wires=2)


@qml.qnode(dev)
def simple_circuit(angle):
    """
    In this function:
        * Prepare the Bell state |Phi+>.
        * Rotate the first qubit around the y-axis by angle
        * Measure the tensor product observable Z0xZ1.

    Args:
        angle (float): how much to rotate a state around the y-axis.

    Returns:
        Union[tensor, float]: the expectation value of the Z0xZ1 observable.
    """
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    qml.RY(angle, 0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angle = json.loads(test_case_input)
    output = simple_circuit(angle).numpy()

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4), "Not the right expectation value"


# These are the public test cases
test_cases = [
    ('1.23456', '0.3299365180851774'),
    ('1.86923', '-0.2940234756205866')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
import json
import pennylane as qml
import pennylane.numpy as np


dev = qml.device('default.qubit', wires=1)


@qml.qnode(dev)
def simple_circuit(angle):
    """
    In this function:
        * Rotate the qubit around the y-axis by angle
        * Measure the expectation value of the Pauli X observable

    Args:
        angle (float): how much to rotate a state around the y-axis

    Returns:
        Union[tensor, float]: The expectation value of the Pauli X observable
    """
    qml.RY(angle, 0)
    return qml.expval(qml.PauliX(0))


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angle = json.loads(test_case_input)
    output = simple_circuit(angle).numpy()

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('1.23456', '0.9440031218347901'),
    ('2.957', '0.1835461227247332')
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

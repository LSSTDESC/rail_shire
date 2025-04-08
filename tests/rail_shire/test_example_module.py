from rail import dsps_fors2_pz
from rail.dsps_fors2_pz import example_module


def test_greetings() -> None:
    """Verify the output of the `greetings` function"""
    output = example_module.greetings()
    assert output == "Hello from LINCC-Frameworks!"


def test_meaning() -> None:
    """Verify the output of the `meaning` function"""
    output = example_module.meaning()
    assert output == 42


def test_json_to_inputs() -> None:
    """Verify the output of the `json_to_inputs` function"""
    output = dsps_fors2_pz.json_to_inputs('src/rail/dsps_fors2_pz/data/defaults.json')
    assert isinstance(output, dict)

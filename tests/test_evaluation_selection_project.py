from click.testing import CliRunner  # type: ignore
import pytest  # type: ignore
from evaluation_selection_project.train import train  # type: ignore


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    """It fails when test split ratio is less than 2."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            "-1",
        ],
    )
    assert result.exit_code == 1


def test_success_for_correct_test_split_ratio(runner: CliRunner) -> None:
    """It success when test split ratio is more than 2."""
    result = runner.invoke(
        train,
        [
            "--max-iter",
            "1000",
            "--test-split-ratio",
            "3",
        ],
    )
    assert result.exit_code == 0

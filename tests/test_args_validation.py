import pytest
from unittest.mock import Mock, patch
from aegis.args_validation import validate_args


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_weights_mismatch(mock_exists):
    args = Mock(
        command="sim",
        config_path=None,
        pickle_path=["a.pkl"],
        pickle_weights=[0.4, 0.8],
    )
    parser = Mock()

    with pytest.raises(ValueError, match=r"Number of pickle weights \(2\) must match number of pickle paths \(1\)"):
        validate_args(args, parser.error)


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_more_paths_than_weights(mock_exists):
    args = Mock(
        command="sim",
        config_path=None,
        pickle_path=["a.pkl", "b.pkl", "c.pkl"],
        pickle_weights=[0.6, 0.5],
    )
    parser = Mock()

    with pytest.raises(ValueError, match=r"Number of pickle weights \(2\) must match number of pickle paths \(3\)"):
        validate_args(args, parser.error)


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_negative_weights(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["a.pkl"], pickle_weights=[-0.5]
    )
    parser = Mock()

    with pytest.raises(ValueError, match=r"Pickle weights must be in range \(0, 1\]"):
        validate_args(args, parser.error)


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_zero_weights(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["a.pkl"], pickle_weights=[0.0]
    )
    parser = Mock()

    with pytest.raises(ValueError, match=r"Pickle weights must be in range \(0, 1\]"):
        validate_args(args, parser.error)


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_weights_greater_than_one(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["a.pkl"], pickle_weights=[1.5]
    )
    parser = Mock()

    with pytest.raises(ValueError, match=r"Pickle weights must be in range \(0, 1\]"):
        validate_args(args, parser.error)


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_valid_weights(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["a.pkl", "b.pkl"], pickle_weights=[0.5, 1.0]
    )
    parser = Mock()

    validate_args(args, parser.error)  # Should not raise


@patch("aegis.args_validation.pathlib.Path.exists", return_value=True)
def test_validate_no_weights(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["a.pkl"], pickle_weights=[]
    )
    parser = Mock()

    validate_args(args, parser.error)  # Should not raise


@patch("aegis.args_validation.pathlib.Path.exists", return_value=False)
def test_validate_missing_pickle_file(mock_exists):
    args = Mock(
        command="sim", config_path=None, pickle_path=["missing.pkl"], pickle_weights=[]
    )
    parser = Mock()

    with pytest.raises(ValueError, match="Pickle file not found: missing.pkl"):
        validate_args(args, parser.error)

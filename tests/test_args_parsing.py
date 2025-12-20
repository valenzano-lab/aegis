import pytest
from aegis.args_parsing import AegisArgumentParser


def test_pickle_path_single():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim", "-p", "test.pkl"])
    assert args.pickle_path == ["test.pkl"]


def test_pickle_path_multiple():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim", "-p", "test1.pkl", "test2.pkl"])
    assert args.pickle_path == ["test1.pkl", "test2.pkl"]


def test_pickle_path_none():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim"])
    assert args.pickle_path == []


def test_pickle_path_empty():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim", "-p"])
    assert args.pickle_path == []


def test_pickle_weights_single():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim", "--pickle_weights", "0.5"])
    assert args.pickle_weights == [0.5]


def test_pickle_weights_multiple():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim", "--pickle_weights", "0.3", "0.7"])
    assert args.pickle_weights == [0.3, 0.7]


def test_pickle_weights_none():
    parser = AegisArgumentParser()
    args = parser.parse_args(["sim"])
    assert args.pickle_weights == []

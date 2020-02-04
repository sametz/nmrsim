import pathlib

from nmrsim.importlib_r import findbin


def test_findbin(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('nmrsim', 'bin'))
    print('test bin is ', test_bin)
    expected_bin = findbin()
    print('expected bin is ', expected_bin)
    assert not expected_bin.exists()
    print('test creating fakefs ', test_bin)
    fs.create_dir(test_bin)
    assert expected_bin.exists()

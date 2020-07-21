# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


def pytest_generate_tests(metafunc):
    if "db" in metafunc.fixturenames:
        metafunc.parametrize("db", ["d1", "d2", "d1"], indirect=True)


class DB1:
    def __init__(self):
        print("DB1")


class DB2:
    def __init__(self):
        print("DB2")


@pytest.fixture
def db(request):
    if request.param == "d1":
        return DB1()
    elif request.param == "d2":
        return DB2()
    else:
        raise ValueError("invalid internal test config")


def test_foo(db):
    assert 1


if __name__ == "__main__":
    pytest.main()

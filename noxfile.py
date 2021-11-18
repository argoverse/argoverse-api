"""Test automation using `nox`."""
import nox
from nox import Session


def _setup(session: Session) -> None:
    """Install AV2 into a virtual environment.

    Args:
        session (Session): `nox` session.
    """
    session.install(".")


@nox.session
def black(session: Session) -> None:
    """Run `black` against AV2.

    Args:
        session (Session): `nox` session.
    """
    _setup(session)
    session.install("black")
    session.run("black", ".")


@nox.session
def isort(session: Session) -> None:
    """Run `isort` against AV2.

    Args:
        session (Session): `nox` session.
    """
    _setup(session)
    session.install("isort")
    session.run("isort", ".")


@nox.session
def mypy(session: Session) -> None:
    """Run `mypy` against AV2.

    Args:
        session (Session): `nox` session.
    """
    _setup(session)
    session.install("mypy")
    session.run("mypy", ".")


@nox.session
def pytest(session: Session) -> None:
    """Run `pytest` against AV2.

    Args:
        session (Session): `nox` session.
    """
    _setup(session)
    session.install("pytest")
    session.run("pytest")

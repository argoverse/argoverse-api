"""Test automation using `nox`."""
import nox
from nox import Session


def _setup(session: Session) -> None:
    session.install(".")


@nox.session
def black(session: Session) -> None:
    _setup(session)
    session.install("black")
    session.run("black", ".")


@nox.session
def isort(session: Session) -> None:
    _setup(session)
    session.install("isort")
    session.run("isort", ".")


@nox.session
def tests(session: Session) -> None:
    _setup(session)
    session.install("pytest")
    session.run("pytest")


@nox.session
def mypy(session: Session) -> None:
    _setup(session)
    session.install("mypy")
    session.run("mypy", ".")

"""Test automation using `nox`."""
import nox
from nox import Session


def _setup(session: Session) -> None:
    session.install(".")


@nox.session
def tests(session: Session) -> None:
    _setup(session)
    session.install("pytest")
    session.run("pytest")


# @nox.session
# def lint(session: Session) -> None:
#     _setup(session)
#     session.install("flake8")
#     session.run("flake8", "")

# @nox.session
# def mypy(session: Session) -> None:
#     _setup(session)
#     session.install("mypy")
#     session.run("mypy")

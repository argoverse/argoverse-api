"""Nox sessions."""
import tempfile
from typing import Any

import nox
from nox.sessions import Session

package = "argoverse"
# nox.options.sessions = "lint", "safety", "mypy", "tests"
nox.options.sessions = "black", "lint", "mypy", "tests"
locations = "argoverse", "tests", "noxfile.py"


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.conda_install("black")
    session.run("black", "-l 120", *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.conda_install(
        "--channel=conda-forge",
        "flake8",
        # "flake8-annotations",
        # "flake8-bandit",
        # "flake8-black",
        # "flake8-bugbear",
        # "flake8-docstrings",
        # "flake8-import-order",
        # "darglint",
    )

    session.run("flake8", *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.conda_install("--channel=conda-forge", "safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.conda_install("mypy")
    session.run("mypy", *args)


# @nox.session(python=["3.7", "3.8"], venv_backend="conda")
# def pytype(session: Session) -> None:
#     """Type-check using pytype."""
#     args = session.posargs or ["--disable=import-error", *locations]
#     session.conda_install("pytype")
#     session.run("pytype", *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run(
        "conda",
        "env",
        "update",
        "--prefix",
        session.virtualenv.location,
        "--file",
        ".conda/environment.yaml",
        # options
        silent=False,
    )
    args = session.posargs or ["--cov", "-m", "not e2e"]
    session.conda_install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock")
    session.run("pytest", *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    args = session.posargs or ["-m", "not e2e"]
    session.conda_install("pytest", "pytest-mock", "typeguard")
    session.run("pytest", f"--typeguard-packages={package}", *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.conda_install("xdoctest")
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def coverage(session: Session) -> None:
    """Upload coverage data."""
    session.conda_install("coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python=["3.7", "3.8"], venv_backend="conda")
def docs(session: Session) -> None:
    """Build the documentation."""
    session.conda_install("sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")

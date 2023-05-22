import nox


@nox.session
def tests(session):
    session.install("-e", ".[test]")
    session.run("pytest", "-v", "tests")


@nox.session
def doctest(session):
    session.install("-e", ".[test,ncx2]")
    session.run(
        "python",
        "-m",
        "doctest",
        "-o",
        "ELLIPSIS",
        "-o",
        "NORMALIZE_WHITESPACE",
        "-v",
        "README.md",
    )


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")

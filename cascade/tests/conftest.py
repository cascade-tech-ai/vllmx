"""Local *joev* test configuration for smart-prediction helpers.

We duplicate the *pytest_addoption* hook here because the top-level test
suite lives in a sibling directory (``tests/``).  When developers run a
single *joev* test file directly, e.g.

    pytest joev/tests/test_smart_prediction_snake_game.py --look-ahead 64

Pytest will *not* import ``tests/conftest.py`` and therefore the custom CLI
flags would be unknown.  Placing this minimal hook alongside the tests that
need it guarantees the options are always registered regardless of which test
subset is executed.
"""


def pytest_addoption(parser):  # noqa: D401 – pytest hook name is fixed
    group = parser.getgroup("smart-prediction options (joev)")
    group.addoption(
        "--look-ahead",
        metavar="N",
        default="32",
        help="Token look-ahead used by smart-prediction simulations (default: 32)",
    )
    group.addoption(
        "--predictor",
        metavar="NAME",
        default="line_diff",
        help="Advanced predictor to evaluate (line_diff, greedy_suffix, naive_sequential).",
    )

def pytest_addoption(parser):
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that use external provider APIs when present.",
    )

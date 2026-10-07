"""Compatibility forwarding for the browser moved to twop_analysis."""


def run_gui(cells_path: str | None = None, background_path: str | None = None):
    from twop_analysis.cell_browser.app import run_gui as launch

    return launch(cells_path=cells_path, background_path=background_path)


def main(argv: list[str] | None = None) -> None:
    from twop_analysis.cell_browser.app import main as launch

    launch(argv)


if __name__ == "__main__":
    main()

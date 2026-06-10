from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def run_gui(cells_path: str | None = None, background_path: str | None = None) -> None:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow(cells_path=cells_path, background_path=background_path)
    window.showMaximized()

    if owns_app:
        sys.exit(app.exec())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visual Cortex Cell Browser")
    parser.add_argument("--cells", default=None, help="Path to .cellDB_pickle file")
    parser.add_argument("--background", default=None, help="Path to .npy background image")
    args = parser.parse_args(argv)
    run_gui(cells_path=args.cells, background_path=args.background)


if __name__ == "__main__":
    main()

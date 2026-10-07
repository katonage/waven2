import os
from importlib.metadata import distribution
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"))

import numpy as np
import pandas as pd
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

from twop_analysis.cell_browser.data_model import CellDataModel
from twop_analysis.cell_browser.main_window import MainWindow
from waven2.cell_browser_plugin import WavenBrowserPlugin
from waven2.cell_browser_plugin.data_view import WavenDataView


class WavenPluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.settings = QSettings(str(self.folder / "settings.ini"), QSettings.IniFormat)
        self.cells = pd.DataFrame({
            "cell_id": [42, 7, 101], "Soma_Xpix": [2., 10., 20.],
            "Soma_Ypix": [3., 11., 21.], "SNR": [1., 2., 3.],
            "CNN": [.1, .8, .9], "Accepted": [True, False, True],
            "Repeatability": [.2, .6, .9], "r_train": [.1, .3, .8],
            "r_test": [.05, .2, .6],
        })
        self.metadata = {
            "SeriesID": "test", "Resolution_um": [.5, .75], "target_fps": 12.,
            "wavelet_params": {"visual_coverage": [-15., 105., -17.5, 50.]},
        }
        for label, best, tuning, axis in (
            ("x", "Azimuth", "tun_xs", "xs"),
            ("y", "Elevation", "tun_ys", "ys"),
            ("angle", "Angle", "tun_angles", "angles"),
            ("size", "Size", "tun_sizes", "sizes"),
            ("freq", "Frequency", "tun_freqs", "freqs"),
            ("phase", "Phase", "tun_phases", "phases"),
        ):
            self.cells[best] = [1., 2., 3.]
            self.cells[tuning] = [np.array([.1, .2, .5, .3, .1])] * 3
            self.metadata["wavelet_params"][axis] = np.arange(5, dtype=float)
        self.cells["tun_xy"] = [np.arange(20).reshape(4, 5)] * 3
        self.cells["Cell_activity"] = [np.arange(30).reshape(3, 10)] * 3
        self.cells["WL_transient_mod"] = [np.arange(10)] * 3
        critical = patch("twop_analysis.cell_browser.main_window.QMessageBox.critical")
        warning = patch("twop_analysis.cell_browser.main_window.QMessageBox.warning")
        self.critical = critical.start()
        self.warning = warning.start()
        self.addCleanup(critical.stop)
        self.addCleanup(warning.stop)

    def database(self, name="renamed.cellDB_pickle", cells=None, metadata=None):
        path = self.folder / name
        with path.open("wb") as handle:
            pickle.dump(self.cells if cells is None else cells, handle)
            pickle.dump(self.metadata if metadata is None else metadata, handle)
        return path

    def window(self, discover=False):
        kwargs = {} if discover else {"plugins": [WavenBrowserPlugin()]}
        window = MainWindow(settings=self.settings, **kwargs)

        def cleanup():
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

        self.addCleanup(cleanup)
        return window

    def test_installed_plugin_is_discovered_and_selected_automatically(self):
        window = self.window(discover=True)
        window.open_cells(str(self.database()))
        self.assertEqual(window.plugin_errors, [])
        self.assertEqual(window.active_plugin.id, "waven2")
        self.assertNotIn("Could not", window.statusBar().currentMessage())
        self.critical.assert_not_called()

    def test_commands_belong_to_framework_and_legacy_cli_forwards(self):
        commands = {
            entry.name: entry.value for entry in distribution("twop_analysis").entry_points
            if entry.group == "console_scripts"
        }
        self.assertEqual(commands["cb"], "twop_analysis.cell_browser.app:main")
        self.assertEqual(commands["cell-browser"], commands["cb"])
        self.assertFalse(any(
            entry.name in {"cb", "cell-browser"} and entry.group == "console_scripts"
            for entry in distribution("waven2").entry_points
        ))
        for module in ("twop_analysis.cell_browser", "waven2.visual_cortex_cell_browser"):
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("--cells", result.stdout)
            self.assertIn("Cell database browser", result.stdout)

    def test_waven_results_keep_selected_statistics_and_temporal_views(self):
        window = self.window()
        window.open_cells(str(self.database()))
        self.critical.assert_not_called()
        self.warning.assert_not_called()
        self.assertEqual(window.active_plugin.id, "waven2")
        self.assertNotIn("Could not", window.statusBar().currentMessage())
        widget = window.plugin_widget
        self.assertEqual(len(widget.figure.axes), 9)  # Eight panels and RF colorbar.
        temporal = widget.temporal_panel.temporal_view
        self.assertEqual(len(temporal.listDataItems()), 4)  # Three trials and their mean.
        x, mean = temporal.listDataItems()[-1].getData()
        np.testing.assert_allclose(x, np.arange(10) / 12.)
        np.testing.assert_allclose(mean, np.arange(30).reshape(3, 10).mean(axis=0))
        self.assertEqual(len(temporal.RightViewBox.addedItems), 1)
        window.select_cell(1)
        self.assertEqual(widget.selected_iloc, 1)
        window.mode_combo.setCurrentText("Stat")
        self.assertTrue(widget.temporal_panel.isHidden())
        self.assertEqual(len(widget.figure.axes), 10)
        window.filter_spins["Repeatability"].setValue(.7)
        self.assertEqual(window.model.filtered_indices.tolist(), [2])
        window.mode_combo.setCurrentText("Selected")
        self.assertFalse(widget.temporal_panel.isHidden())
        self.assertEqual(widget.selected_iloc, 2)
        window.filter_spins["Repeatability"].setValue(.95)
        self.assertIsNone(widget.selected_iloc)
        self.assertNotIn("Could not", window.statusBar().currentMessage())

    def test_caiman_database_uses_plain_even_with_waven_provider_installed(self):
        window = self.window()
        caiman = self.cells[["cell_id", "Soma_Xpix", "Soma_Ypix", "SNR", "CNN", "Accepted"]]
        metadata = {"SeriesID": "caiman", "Resolution_um": [.5, .75], "cnm_framerate": 10.}
        window.open_cells(str(self.database("cells_caiman.cellDB_pickle", caiman, metadata)))
        self.assertEqual(window.active_plugin.id, "plain")
        self.assertEqual(set(window.filter_spins), {"SNR", "CNN"})
        self.assertNotIn("x", window.display_fields)
        self.critical.assert_not_called()
        self.warning.assert_not_called()

    def test_legacy_waven_columns_match_without_wavelet_metadata(self):
        model = CellDataModel(df=self.cells, metadata={})
        self.assertTrue(WavenBrowserPlugin().matches(model))
        view = WavenDataView(model)
        self.assertIsNone(view.wavelet_axis("xs"))
        self.assertIsNone(view.visual_coverage)

    def test_other_analysis_tuning_does_not_select_waven_plugin(self):
        cells = pd.DataFrame({"Soma_Xpix": [1.], "Soma_Ypix": [2.], "tun_response": [[1., 2.]]})
        model = CellDataModel(df=cells, metadata={})
        self.assertFalse(WavenBrowserPlugin().matches(model))

    def test_waven_metadata_is_not_carried_between_databases(self):
        window = self.window()
        window.open_cells(str(self.database()))
        self.assertEqual(window.plugin_widget.model.visual_coverage, [-15., 105., -17.5, 50.])
        metadata = {"SeriesID": "legacy", "Resolution_um": [.5, .75], "target_fps": 15.}
        window.open_cells(str(self.database("legacy.pkl", metadata=metadata)))
        self.assertEqual(window.active_plugin.id, "waven2")
        self.assertIsNone(window.plugin_widget.model.visual_coverage)
        self.assertIsNone(window.plugin_widget.model.wavelet_axis("xs"))
        self.assertEqual(window.plugin_widget.model.target_fps, 15.)
        self.assertNotIn("Could not", window.statusBar().currentMessage())
        self.critical.assert_not_called()


if __name__ == "__main__":
    unittest.main()

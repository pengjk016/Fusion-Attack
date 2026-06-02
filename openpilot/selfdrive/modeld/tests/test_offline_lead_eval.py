#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from openpilot.selfdrive.modeld.offline_lead_eval import ImageDirectoryFrameReader, metric_stats


class TestOfflineLeadEval(unittest.TestCase):
  def test_numeric_frame_ids(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      image_dir = Path(tmp_dir)
      img = np.full((4, 6, 3), 120, dtype=np.uint8)
      Image.fromarray(img).save(image_dir / "10.png")
      Image.fromarray(img).save(image_dir / "11.png")

      reader = ImageDirectoryFrameReader(image_dir, 6, 4, None)
      nv12 = reader.get(10, pix_fmt="nv12")[0]

      self.assertEqual(nv12.shape, (6 * 4 * 3 // 2,))

  def test_sequential_frame_ids_with_offset(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      image_dir = Path(tmp_dir)
      dark = np.zeros((4, 4, 3), dtype=np.uint8)
      bright = np.full((4, 4, 3), 255, dtype=np.uint8)
      Image.fromarray(dark).save(image_dir / "frame_a.png")
      Image.fromarray(bright).save(image_dir / "frame_b.png")

      reader = ImageDirectoryFrameReader(image_dir, 4, 4, 7)
      rgb_dark = reader.get(7, pix_fmt="rgb24")[0]
      rgb_bright = reader.get(8, pix_fmt="rgb24")[0]

      self.assertLess(np.mean(rgb_dark), np.mean(rgb_bright))

  def test_metric_stats(self):
    stats = metric_stats([1.0, -1.0, 3.0, -3.0])

    self.assertEqual(stats["count"], 4)
    self.assertAlmostEqual(stats["mean_signed"], 0.0)
    self.assertAlmostEqual(stats["mae"], 2.0)
    self.assertAlmostEqual(stats["rmse"], np.sqrt(5.0))
    self.assertAlmostEqual(stats["max_abs"], 3.0)


if __name__ == "__main__":
  unittest.main()

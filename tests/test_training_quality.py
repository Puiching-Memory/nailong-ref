from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from nailong import manifests
from nailong_tts import dataset, quality, synth


class TrainingQualityTest(unittest.TestCase):
    def test_synthesis_requires_explicit_reviewed_reference(self) -> None:
        with patch.object(synth, "REF_WAV", None), patch.object(synth, "REF_TEXT", None):
            with self.assertRaisesRegex(ValueError, "已听音核对"):
                synth.Backend()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio_dir = self.root / "production"
        self.wav = self.audio_dir / "accepted" / "sample.wav"
        self.wav.parent.mkdir(parents=True)
        t = np.arange(24000 * 3) / 24000
        sf.write(self.wav, 0.2 * np.sin(2 * np.pi * 250 * t), 24000, subtype="PCM_16")
        self.manifest = self.root / "production.csv"
        self.review = self.root / "review.csv"
        self.hash = quality.sha256(self.wav)
        manifests.write(self.manifest,
                        ["file", "status", "dur", "speaker_basis", "stem_margin_db",
                         "text", "sha256"],
                        [["accepted/sample.wav", "accepted", "3.00", "visual_confirmed",
                          "20.0", "你好。😊", self.hash]])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_approval_is_bound_to_audio_and_corrected_text_survives_refresh(self) -> None:
        quality.make_review(self.manifest, self.review)
        with self.review.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["text"], "你好。")
        row["approved"] = "yes"
        row["text"] = "您好。"
        manifests.write(self.review, quality.FIELDS, [[row[field] for field in quality.FIELDS]])
        quality.make_review(self.manifest, self.review)
        good = quality.audit(self.manifest, self.audio_dir, self.review, 0, 0)
        self.assertFalse(good.issues)
        self.assertEqual(good.usable[0][1], "您好。")

        # A changed waveform invalidates both the manifest hash and the approval.
        sf.write(self.wav, np.zeros(24000 * 3), 24000, subtype="PCM_16")
        bad = quality.audit(self.manifest, self.audio_dir, self.review, 0, 0)
        self.assertTrue(any("hash_mismatch" in issue for issue in bad.issues))

    def test_quantity_gate_blocks_tiny_corpus(self) -> None:
        quality.make_review(self.manifest, self.review)
        result = quality.audit(self.manifest, self.audio_dir, self.review)
        self.assertTrue(any("corpus:" in issue for issue in result.issues))

    def test_failed_pack_removes_stale_filelist(self) -> None:
        # pack uses the repository review path, so this tests the missing-review case.
        out = self.root / "build"
        out.mkdir()
        (out / "filelist.txt").write_text("stale", encoding="utf-8")
        with self.assertRaises(SystemExit):
            dataset.pack(out, self.manifest, self.audio_dir)
        self.assertFalse((out / "filelist.txt").exists())

    def test_experimental_pack_uses_reviewed_transcript_and_provenance(self) -> None:
        quality.make_review(self.manifest, self.review)
        manifests.write(self.review, quality.FIELDS,
                        [["accepted/sample.wav", self.hash, "yes", "您好。", "checked"]])
        out = self.root / "build"
        dataset.pack(out, self.manifest, self.audio_dir,
                     experimental=True, review=self.review)
        filelist = (out / "filelist.txt").read_text(encoding="utf-8")
        self.assertIn("|nailong|zh|您好。", filelist)
        with (out / "metadata.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["sha256"], self.hash)
        self.assertEqual(row["text"], "您好。")


if __name__ == "__main__":
    unittest.main()

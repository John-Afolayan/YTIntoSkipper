"""Offline regressions using the repository's real reference and speech audio.

No downloader, SponsorBlock client or production database is instantiated.
Run with: python -m unittest test_audio_detection
"""
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_fingerprint import AudioFingerprinter, AudioAnalysisError, _get_ffmpeg_executable, get_audio_duration
from automator import IntroSkipperAutomator

ROOT = Path(__file__).resolve().parent


def detector(fingerprinter, reference, search_limit=120):
    app = IntroSkipperAutomator.__new__(IntroSkipperAutomator)
    app.fingerprinter = fingerprinter
    app._references = [reference]
    app.peak_height = .25
    app.weighted_threshold = .60
    app.search_limit = search_limit
    app.early_exit_threshold = .90
    app.trim_speech = False
    return app


class AudioDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fp = AudioFingerprinter()
        cls.ref = cls.fp.generate_fingerprint(str(ROOT / "intro.m4a"), use_cache=False)
        cls.sr = cls.fp.sample_rate
        cls.intro, _, tmp = cls.fp._load_audio(str(ROOT / "intro.m4a"))
        if tmp:
            os.unlink(tmp)
        cls.speech, _, tmp = cls.fp._load_audio(str(ROOT / "test.m4a"), offset=30, duration=20)
        if tmp:
            os.unlink(tmp)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write(self, name, audio):
        path = str(Path(self.tmp.name) / (name + ".wav"))
        sf.write(path, audio, self.sr, subtype="FLOAT")
        return path

    def detect(self, audio, name="video", search_limit=120):
        return detector(self.fp, self.ref, search_limit).find_intro_in_video(self.write(name, audio))

    def test_known_offsets_with_speech_leadin_and_preserved_buffer(self):
        for offset in [0, .12, .35, .8, 12.037, 60, 105]:
            with self.subTest(offset=offset):
                lead = np.resize(self.speech, round(offset * self.sr))
                segment = self.detect(np.concatenate([lead, self.intro, self.speech[:self.sr]]))
                self.assertIsNotNone(segment)
                expected = offset - .2 if offset > .5 else offset
                self.assertAlmostEqual(segment.start_time, expected, delta=.08)
                self.assertAlmostEqual(segment.end_time, offset + self.ref["duration"], delta=.15)

    def test_reference_only_endpoint_is_detected(self):
        segment = self.detect(self.intro)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.start_time, 0)

    def test_match_at_last_valid_scan_position(self):
        lead = np.zeros(round(9.317 * self.sr))
        audio = np.concatenate([lead, self.intro])
        segment = self.detect(audio, search_limit=len(audio) / self.sr)
        self.assertIsNotNone(segment)
        self.assertAlmostEqual(segment.start_time, 9.117, delta=.08)

    def test_gain_noise_and_aac_reencoding(self):
        rng = np.random.default_rng(42)
        changed = .25 * self.intro + rng.normal(0, .0005, len(self.intro))
        segment = self.detect(np.concatenate([np.zeros(self.sr * 4), changed, np.zeros(self.sr)]))
        self.assertIsNotNone(segment)
        self.assertAlmostEqual(segment.start_time, 3.8, delta=.08)
        wav = self.write("codec_input", np.concatenate([
            np.zeros(self.sr * 5), self.intro, np.zeros(self.sr)]))
        encoded = str(Path(self.tmp.name) / "encoded.m4a")
        subprocess.run([_get_ffmpeg_executable(), "-y", "-v", "error", "-i", wav,
                        "-ar", "44100", "-c:a", "aac", "-b:a", "64k", encoded],
                       check=True, capture_output=True)
        segment = detector(self.fp, self.ref).find_intro_in_video(encoded)
        self.assertIsNotNone(segment)
        self.assertAlmostEqual(segment.start_time, 4.8, delta=.08)

    def test_quiet_intro_after_loud_speech(self):
        audio = np.concatenate([self.speech, self.intro * .02, self.speech[:self.sr]])
        segment = self.detect(audio)
        self.assertIsNotNone(segment)
        self.assertAlmostEqual(segment.start_time, len(self.speech) / self.sr - .2, delta=.08)

    def test_no_intro_negatives(self):
        rng = np.random.default_rng(7)
        time = np.arange(len(self.intro) * 2) / self.sr
        chord = sum(.05 * np.sin(2 * np.pi * hz * time) for hz in [220, 277.18, 329.63])
        shuffled = np.concatenate(list(reversed(np.array_split(self.intro, 10))))
        cases = {"silence": np.zeros(self.sr * 20), "speech": self.speech,
                 "noise": rng.normal(0, .03, self.sr * 20), "chord": chord,
                 "reversed": self.intro[::-1], "shuffled": shuffled,
                 "short_excerpt": np.concatenate([self.intro[:self.sr * 3], self.speech])}
        for name, audio in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(self.detect(audio, name))

    def test_repeated_intro_uses_first_verified_occurrence(self):
        audio = np.concatenate([self.intro, self.speech, self.intro, self.speech[:self.sr]])
        segment = self.detect(audio)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.start_time, 0)

    def test_short_video_cannot_verify_full_intro(self):
        path = self.write("truncated", self.intro[:self.sr * 4])
        self.assertIsNone(self.fp.verify_match_quality(self.ref, path, 0))
        self.assertIsNone(detector(self.fp, self.ref).find_intro_in_video(path))

    def test_silent_reference_is_rejected(self):
        path = self.write("silent_reference", np.zeros(self.sr * 3))
        self.assertIsNone(self.fp.generate_fingerprint(path, use_cache=False))

    def test_known_talkover_never_auto_submits(self):
        segment = detector(self.fp, self.ref).find_intro_in_video(str(ROOT / "test.m4a"))
        self.assertTrue(segment is None or segment.force_review)

    def test_tail_speech_retains_detection(self):
        audio = self.intro.copy()
        n = int(2.5 * self.sr)
        speech = self.speech[:n]
        speech = speech * (np.sqrt(np.mean(audio[-n:] ** 2)) /
                           (np.sqrt(np.mean(speech ** 2)) + 1e-9))
        audio[-n:] += speech
        segment = self.detect(audio)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.start_time, 0)

    def test_verification_uses_scan_features_without_decoding_again(self):
        path = self.write("reuse", self.intro)
        self.fp.scan_video(self.ref, path)
        with patch.object(self.fp, "_load_audio", side_effect=AssertionError("duplicate decode")):
            self.assertTrue(self.fp.verify_match_quality(self.ref, path, 0)["verified"])

    def test_same_path_replaced_audio_does_not_reuse_stale_scan(self):
        path = self.write("replaced", self.intro)
        self.fp.scan_video(self.ref, path)
        self.write("replaced", self.speech)
        self.assertFalse(self.fp.verify_match_quality(self.ref, path, 0)["verified"])

    def test_cache_version_and_sample_rate_and_round_trip(self):
        path = str(ROOT / "intro.m4a")
        self.assertNotEqual(self.fp._cache_key(path, None, 0, 22050),
                            self.fp._cache_key(path, None, 0, 16000))
        with patch.object(AudioFingerprinter, "CACHE_DIR", Path(self.tmp.name) / "cache"):
            key = self.fp._cache_key(path, None, 0)
            self.fp._save_cache(key, self.ref)
            cached = self.fp._load_cached(key)
            np.testing.assert_array_equal(cached["spectral"], self.ref["spectral"])
            with patch.object(self.fp, "_load_audio", side_effect=AssertionError("cache miss")):
                self.assertIsNotNone(self.fp.generate_fingerprint(path))

    def test_custom_sample_rate_is_consistent(self):
        fp = AudioFingerprinter(sample_rate=16000)
        ref = fp.generate_fingerprint(str(ROOT / "intro.m4a"), use_cache=False)
        self.assertEqual(ref["sr"], 16000)
        self.assertIsNotNone(detector(fp, ref).find_intro_in_video(str(ROOT / "intro.m4a")))

    def test_duration_works_without_ffprobe(self):
        with patch("audio_fingerprint.shutil.which", return_value=None):
            self.assertAlmostEqual(get_audio_duration(str(ROOT / "intro.m4a")),
                                   self.ref["duration"], delta=.1)

    def test_multiple_references_select_matching_intro(self):
        unrelated = self.write("other_reference", self.intro[::-1])
        other = self.fp.generate_fingerprint(unrelated, use_cache=False)
        app = detector(self.fp, self.ref)
        app._references = [other, self.ref]
        path = self.write("multi_reference_video", np.concatenate([
            self.speech[:self.sr * 5], self.intro, self.speech[:self.sr]]))
        segment = app.find_intro_in_video(path)
        self.assertIsNotNone(segment)
        self.assertAlmostEqual(segment.start_time, 4.8, delta=.08)

    def test_decode_error_is_not_reported_as_no_intro(self):
        with patch.object(self.fp, "_load_audio", side_effect=RuntimeError("decoder failed")):
            with self.assertRaises(AudioAnalysisError):
                self.fp.scan_video(self.ref, "missing.wav")

    def test_speech_trim_requires_positive_confirmation(self):
        audio = self.intro.copy()
        audio[-self.sr * 3:] += self.speech[:self.sr * 3] * 5
        path = self.write("unconfirmed_speech", audio)
        with patch.object(self.fp, "_vad_confirms_speech", return_value=None), \
             patch.object(self.fp, "_modulation_confirms_speech", return_value=None):
            self.assertIsNone(self.fp.detect_speech_overlay(self.ref, path, 0))

    def test_shared_fingerprinter_keeps_workers_isolated(self):
        def run(offset):
            path = self.write("worker" + str(offset),
                              np.concatenate([np.zeros(self.sr * offset), self.intro]))
            app = detector(self.fp, self.ref)
            segment = app.find_intro_in_video(path)
            return segment.start_time if segment else None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, [3, 7]))
        for result, expected in zip(results, [2.8, 6.8]):
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, expected, delta=.08)


if __name__ == "__main__":
    unittest.main()

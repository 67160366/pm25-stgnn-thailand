"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Tests for app/lib/telegram.py (Telegram Bot API alert formatting + sending).
All tests are offline: HTTP is mocked with requests-mock. ``live_forecast`` is
never called — every test injects its own fake station/prediction data.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import requests

from app.lib import telegram

HORIZONS = [6, 12, 24, 48]


class TestShouldAlert:
    """should_alert() trigger logic."""

    def test_exceeds_threshold(self):
        assert telegram.should_alert([10.0, 20.0, 40.0, 30.0], threshold=37.5) is True

    def test_exactly_at_threshold_triggers(self):
        assert telegram.should_alert([37.5], threshold=37.5) is True

    def test_below_threshold_does_not_trigger(self):
        assert telegram.should_alert([10.0, 20.0, 30.0, 36.0], threshold=37.5) is False

    def test_nan_entries_ignored(self):
        assert telegram.should_alert([float("nan"), float("nan")], threshold=37.5) is False

    def test_nan_mixed_with_exceeding_value_triggers(self):
        assert telegram.should_alert([float("nan"), 50.0], threshold=37.5) is True

    def test_empty_sequence_no_force(self):
        assert telegram.should_alert([], threshold=37.5) is False

    def test_force_overrides_low_values(self):
        assert telegram.should_alert([1.0, 2.0], threshold=37.5, force=True) is True

    def test_force_overrides_empty(self):
        assert telegram.should_alert([], force=True) is True

    def test_default_threshold_matches_aqi_band_boundary(self):
        # DEFAULT_ALERT_THRESHOLD_UG must equal app.lib.aqi's moderate/risk-group
        # band boundary so the alert trigger and the dashboard AQI band agree.
        from app.lib import aqi

        just_below = telegram.DEFAULT_ALERT_THRESHOLD_UG - 0.01
        just_at = telegram.DEFAULT_ALERT_THRESHOLD_UG
        assert aqi.category(just_below) != aqi.category(just_at)
        assert aqi.category(just_at) == "เริ่มมีผลต่อกลุ่มเสี่ยง"


class TestFormatAlert:
    """format_alert() message construction."""

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp("2026-07-11T07:00:00", tz="UTC")  # 14:00 Bangkok

    def test_contains_station_and_province_header(self):
        text = telegram.format_alert(
            "โรงเรียนยุพราชวิทยาลัย",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            self._now(),
        )
        assert "โรงเรียนยุพราชวิทยาลัย (เชียงใหม่)" in text

    def test_omits_parens_when_province_empty(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        assert "สถานีทดสอบ ()" not in text
        assert "สถานีทดสอบ" in text

    def test_bangkok_timestamp_rendered(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        # 07:00 UTC -> 14:00 Asia/Bangkok
        assert "14:00" in text
        assert "11/07/2026" in text

    def test_all_horizons_present(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        for h in HORIZONS:
            assert f"{h} ชม." in text

    def test_current_value_and_aqi_band_shown(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        assert "30.0" in text
        assert "ปานกลาง" in text  # 30.0 ug/m3 falls in the "moderate" band

    def test_missing_observed_shown_as_no_data(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", None, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        assert "ค่าปัจจุบัน: ไม่มีข้อมูล" in text

    def test_nan_observed_shown_as_no_data(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            float("nan"),
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            self._now(),
        )
        assert "ค่าปัจจุบัน: ไม่มีข้อมูล" in text

    def test_halfwidths_shown_when_present(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            self._now(),
            halfwidths=[5.0, 6.0, 7.0, 8.0],
        )
        assert "±5.0" in text
        assert "±8.0" in text

    def test_halfwidths_omitted_when_none(self):
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        assert "±" not in text

    def test_nan_horizon_prediction_shown_as_no_data(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, float("nan"), 40.0, 35.0],
            self._now(),
        )
        assert "12 ชม. — ไม่มีข้อมูล" in text

    def test_all_nan_predictions_no_advice_line_and_no_crash(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [float("nan")] * 4,
            self._now(),
        )
        assert "คำแนะนำ" not in text

    def test_advice_uses_worst_case_horizon(self):
        # Highest predicted value (60.0, band "มีผลต่อสุขภาพ") should drive advice,
        # not the lowest (20.0, band "ปานกลาง").
        text = telegram.format_alert(
            "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 60.0, 35.0], self._now()
        )
        assert "คำแนะนำ (มีผลต่อสุขภาพ)" in text

    def test_live_mode_adds_nwp_caveat(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            self._now(),
            live_mode=True,
        )
        assert "NWP" in text

    def test_non_live_mode_omits_nwp_caveat(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            self._now(),
            live_mode=False,
        )
        assert "NWP" not in text
        assert "MTGNN" in text

    def test_accepts_iso_string_timestamp(self):
        text = telegram.format_alert(
            "สถานีทดสอบ",
            "เชียงใหม่",
            30.0,
            HORIZONS,
            [20.0, 25.0, 40.0, 35.0],
            "2026-07-11T07:00:00+00:00",
        )
        assert "14:00" in text

    def test_mismatched_pred_length_raises(self):
        with pytest.raises(ValueError):
            telegram.format_alert(
                "สถานีทดสอบ", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0], self._now()
            )

    def test_mismatched_halfwidths_length_raises(self):
        with pytest.raises(ValueError):
            telegram.format_alert(
                "สถานีทดสอบ",
                "เชียงใหม่",
                30.0,
                HORIZONS,
                [20.0, 25.0, 40.0, 35.0],
                self._now(),
                halfwidths=[1.0, 2.0],
            )

    def test_special_characters_passed_through_unescaped(self):
        # format_alert emits plain text (no parse_mode), so special HTML/Markdown
        # characters in a station name should NOT be escaped or stripped.
        text = telegram.format_alert(
            "สถานี <Test> & Co.", "เชียงใหม่", 30.0, HORIZONS, [20.0, 25.0, 40.0, 35.0], self._now()
        )
        assert "สถานี <Test> & Co." in text


class TestSendMessage:
    """send_message() HTTP behaviour against the Telegram Bot API."""

    _URL = "https://api.telegram.org/bottest-token/sendMessage"

    def test_success_returns_body_and_posts_expected_payload(self, requests_mock):
        requests_mock.post(
            self._URL, json={"ok": True, "result": {"message_id": 42}}, status_code=200
        )
        body = telegram.send_message("test-token", "12345", "สวัสดี")
        assert body["ok"] is True
        assert requests_mock.last_request.json() == {"chat_id": "12345", "text": "สวัสดี"}

    def test_parse_mode_included_when_given(self, requests_mock):
        requests_mock.post(self._URL, json={"ok": True, "result": {}}, status_code=200)
        telegram.send_message("test-token", "12345", "<b>hi</b>", parse_mode="HTML")
        sent = requests_mock.last_request.json()
        assert sent["parse_mode"] == "HTML"

    def test_parse_mode_omitted_by_default(self, requests_mock):
        requests_mock.post(self._URL, json={"ok": True, "result": {}}, status_code=200)
        telegram.send_message("test-token", "12345", "hi")
        assert "parse_mode" not in requests_mock.last_request.json()

    def test_http_error_status_raises(self, requests_mock):
        requests_mock.post(
            "https://api.telegram.org/botbad-token/sendMessage",
            json={"ok": False, "description": "Unauthorized"},
            status_code=401,
        )
        with pytest.raises(telegram.TelegramSendError, match="Unauthorized"):
            telegram.send_message("bad-token", "12345", "hi")

    def test_ok_false_with_200_status_raises(self, requests_mock):
        requests_mock.post(
            self._URL,
            json={"ok": False, "description": "chat not found"},
            status_code=200,
        )
        with pytest.raises(telegram.TelegramSendError, match="chat not found"):
            telegram.send_message("test-token", "bad-chat", "hi")

    def test_timeout_raises_telegram_send_error(self, requests_mock):
        requests_mock.post(self._URL, exc=requests.exceptions.Timeout)
        with pytest.raises(telegram.TelegramSendError, match="timed out"):
            telegram.send_message("test-token", "12345", "hi")

    def test_connection_error_raises_telegram_send_error(self, requests_mock):
        requests_mock.post(self._URL, exc=requests.exceptions.ConnectionError)
        with pytest.raises(telegram.TelegramSendError):
            telegram.send_message("test-token", "12345", "hi")

    def test_non_json_body_raises_telegram_send_error(self, requests_mock):
        requests_mock.post(self._URL, text="<html>not json</html>", status_code=200)
        with pytest.raises(telegram.TelegramSendError, match="non-JSON"):
            telegram.send_message("test-token", "12345", "hi")

    def test_token_never_leaks_into_error_or_chain(self, requests_mock):
        # requests exception text embeds the URL (which contains the bot token);
        # send_message must redact it and suppress the chained cause.
        requests_mock.post(
            self._URL,
            exc=requests.exceptions.ConnectionError(f"Max retries exceeded with url: {self._URL}"),
        )
        with pytest.raises(telegram.TelegramSendError) as excinfo:
            telegram.send_message("test-token", "12345", "hi")
        assert "test-token" not in str(excinfo.value)
        assert "<redacted-token>" in str(excinfo.value)
        assert excinfo.value.__cause__ is None


def test_default_threshold_is_finite_positive_number():
    assert math.isfinite(telegram.DEFAULT_ALERT_THRESHOLD_UG)
    assert telegram.DEFAULT_ALERT_THRESHOLD_UG > 0

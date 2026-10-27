"""
Tests for reading and writing Session clip settings
(get_clip_info, set_clip_property).

The Remote Script half runs against a fake Song/Clip (no Live); the server
half runs the real tool functions with the socket connection mocked.

Run from the repo root:
    uv run --with pytest pytest -v tests/test_clip_properties.py
"""

import json

import pytest

import MCP_Server.server as server
# The server-side fixtures (mocked connection, telemetry and consent pinned
# off, every capability reported present) are shared with the clip-notes tests.
from test_clip_notes import (  # noqa: F401
    _no_consent_prompt,
    _script_capabilities_available,
    _silence_telemetry,
    call,
    fake_conn,
)
from test_remote_script_bridge import _load_script_module, make_instance


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

class FakeAudioClip(object):
    is_audio_clip = True
    is_midi_clip = False

    def __init__(self):
        self.name = "vocals"
        self.length = 64.0
        self.looping = False
        self.loop_start = 0.0
        self.loop_end = 64.0
        self.start_marker = 0.0
        self.end_marker = 64.0
        self.warping = True
        self.warp_mode = 0
        self.available_warp_modes = [0, 1, 2, 3]
        self.pitch_coarse = 0
        self.pitch_fine = 0.0
        self.gain = 0.4
        self.gain_display_string = "0.0 dB"
        self.file_path = "C:/x/vocals.wav"


class FakeMidiClip(object):
    is_audio_clip = False
    is_midi_clip = True

    def __init__(self):
        self.name = "keys"
        self.length = 4.0
        self.looping = True
        self.loop_start = 0.0
        self.loop_end = 4.0
        self.start_marker = 0.0
        self.end_marker = 4.0


class FakeSlot(object):
    def __init__(self, clip=None):
        self.clip = clip

    @property
    def has_clip(self):
        return self.clip is not None


class FakeTrack(object):
    def __init__(self, slots):
        self.clip_slots = slots


class FakeSong(object):
    def __init__(self, tracks):
        self.tracks = tracks


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


@pytest.fixture
def live(script_module):
    """Script instance over one audio clip (0,0), one MIDI clip (0,1) and an
    empty slot (0,2)."""
    audio, midi = FakeAudioClip(), FakeMidiClip()
    inst = make_instance(script_module)
    inst._song = FakeSong([FakeTrack([FakeSlot(audio), FakeSlot(midi), FakeSlot()])])
    return inst, audio, midi


# --------------------------------------------------------------------------
# Remote Script: get_clip_info
# --------------------------------------------------------------------------

def test_get_clip_info_reads_audio_clip_settings(live):
    inst, _, _ = live
    info = inst._get_clip_info(0, 0)
    assert info["is_audio_clip"] is True
    assert info["start_marker"] == 0.0
    assert info["end_marker"] == 64.0
    assert info["warping"] is True
    assert info["warp_mode"] == 0
    assert info["warp_mode_name"] == "Beats"
    assert info["available_warp_modes"] == [0, 1, 2, 3]
    assert info["pitch_coarse"] == 0
    assert info["pitch_fine"] == 0.0
    assert info["gain"] == 0.4
    assert info["gain_display_string"] == "0.0 dB"
    assert info["file_path"] == "C:/x/vocals.wav"


def test_get_clip_info_keeps_fractional_detune(live):
    """Detune is in cents and Live stores it as a float; don't truncate."""
    inst, audio, _ = live
    audio.pitch_fine = -12.5
    assert inst._get_clip_info(0, 0)["pitch_fine"] == -12.5


def test_get_clip_info_on_midi_clip_omits_audio_fields(live):
    inst, _, _ = live
    info = inst._get_clip_info(0, 1)
    assert info["is_midi_clip"] is True
    assert info["start_marker"] == 0.0
    for key in ("warping", "warp_mode", "pitch_coarse", "gain", "file_path"):
        assert key not in info


def test_get_clip_info_empty_slot_raises(live):
    inst, _, _ = live
    with pytest.raises(Exception, match="No clip"):
        inst._get_clip_info(0, 2)


# --------------------------------------------------------------------------
# Remote Script: set_clip_property
# --------------------------------------------------------------------------

def test_set_pitch_coarse_writes_and_reports_old_value(live):
    inst, audio, _ = live
    result = inst._set_clip_property(0, 0, "pitch_coarse", -2)
    assert audio.pitch_coarse == -2
    assert result["old_value"] == 0
    assert result["value"] == -2


def test_set_start_marker_on_midi_clip_is_allowed(live):
    inst, _, midi = live
    inst._set_clip_property(0, 1, "start_marker", 1.0)
    assert midi.start_marker == 1.0


def test_set_warping_accepts_bool_and_strings(live):
    inst, audio, _ = live
    inst._set_clip_property(0, 0, "warping", False)
    assert audio.warping is False
    inst._set_clip_property(0, 0, "warping", "true")
    assert audio.warping is True


def test_property_outside_allowlist_is_refused(live):
    inst, audio, _ = live
    with pytest.raises(ValueError, match="not writable"):
        inst._set_clip_property(0, 0, "name", "x")
    assert audio.name == "vocals"


def test_audio_only_property_on_midi_clip_is_refused(live):
    inst, _, _ = live
    with pytest.raises(ValueError, match="audio clips only"):
        inst._set_clip_property(0, 1, "pitch_coarse", 2)


def test_fractional_value_for_integer_property_is_refused(live):
    """-2.5 semitones must not silently become -2."""
    inst, audio, _ = live
    with pytest.raises(ValueError, match="whole number"):
        inst._set_clip_property(0, 0, "pitch_coarse", -2.5)
    assert audio.pitch_coarse == 0


def test_set_clip_property_runs_on_main_thread(script_module, live):
    """Writes go through schedule_message like every other state change."""
    inst, audio, _ = live
    scheduled = []

    def schedule_message(_delay, task):
        scheduled.append(task)
        task()

    inst.schedule_message = schedule_message
    response = inst._process_command({
        "type": "set_clip_property",
        "params": {"track_index": 0, "clip_index": 0,
                   "property": "pitch_coarse", "value": 3},
    })
    assert response["status"] == "success"
    assert len(scheduled) == 1
    assert audio.pitch_coarse == 3


def test_new_commands_are_advertised(script_module):
    assert "get_clip_info" in script_module.SCRIPT_CAPABILITIES
    assert "set_clip_property" in script_module.SCRIPT_CAPABILITIES


# --------------------------------------------------------------------------
# Server tools
# --------------------------------------------------------------------------

def test_get_clip_info_tool_sends_command(fake_conn):
    conn = fake_conn(response={"warping": True, "pitch_coarse": 0})
    out = call(server.get_clip_info, None, track_index=2, clip_index=0)
    assert conn.sent == [("get_clip_info", {"track_index": 2, "clip_index": 0})]
    assert json.loads(out)["warping"] is True


def test_set_clip_property_tool_sends_command(fake_conn):
    conn = fake_conn(response={"property": "pitch_coarse",
                               "old_value": 0, "value": -2})
    out = call(server.set_clip_property, None, track_index=2, clip_index=0,
               property="pitch_coarse", value=-2)
    assert conn.sent == [("set_clip_property", {
        "track_index": 2, "clip_index": 0,
        "property": "pitch_coarse", "value": -2})]
    assert "0" in out and "-2" in out


def test_set_clip_property_tool_refuses_unknown_property_locally(fake_conn):
    conn = fake_conn()
    out = call(server.set_clip_property, None, track_index=0, clip_index=0,
               property="name", value=1)
    assert conn.sent == []
    assert "not writable" in out


def test_set_clip_property_gets_the_modifying_command_timeout():
    """Writes wait on Live's main thread, so they get the longer timeout."""
    timeouts = []

    class FakeSocket(object):
        def sendall(self, _data):
            pass

        def settimeout(self, value):
            timeouts.append(value)

    conn = server.AbletonConnection(host="127.0.0.1", port=0)
    conn.sock = FakeSocket()
    conn.receive_full_response = (
        lambda _sock: b'{"status": "success", "result": {}}')
    conn.send_command("set_clip_property", {"property": "gain", "value": 0.4})
    conn.send_command("get_clip_info", {})
    assert timeouts == [15.0, 10.0]

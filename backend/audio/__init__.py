"""
Audio engine for synesthesia BCI system.

Provides real-time audio synthesis with multiple sound options per layer.
"""

from backend.audio.synthesizers import (
    SynthesizerBase,
    SineWaveSynth,
    SquareWaveSynth,
    SawtoothWaveSynth,
    TriangleWaveSynth,
    FMSynth,
    SubtractiveSynth
)
from backend.audio.effects import (
    ReverbEffect,
    DelayEffect,
    FilterEffect,
    CompressorEffect
)
from backend.audio.mixer import AudioMixer, AudioTrack
from backend.audio.engine import AudioEngine

__all__ = [
    'SynthesizerBase',
    'SineWaveSynth',
    'SquareWaveSynth',
    'SawtoothWaveSynth',
    'TriangleWaveSynth',
    'FMSynth',
    'SubtractiveSynth',
    'ReverbEffect',
    'DelayEffect',
    'FilterEffect',
    'CompressorEffect',
    'AudioMixer',
    'AudioTrack',
    'AudioEngine'
]

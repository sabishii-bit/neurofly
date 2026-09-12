"""Talking to the real PC: video and audio in, keyboard and mouse out.

    video.py     VideoSource: ScreenCapture (a window, region or monitor), VideoFile, SyntheticVideo
    audio.py     AudioSource: AudioCapture (a device, or loopback of what the PC plays),
                 AudioFile, SilentAudio, SyntheticAudio
    controls.py  Controls: PCControls (real keyboard and mouse), LoggingControls, NullControls;
                 InputRecorder watches a human; PanicKey stops everything

The optional ``pc`` extra installs the libraries these need (mss, pynput, sounddevice,
soundfile, soundcard); they are imported lazily, so the rest of the package works without.
"""

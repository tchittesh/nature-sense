import acoular as ac
import numpy as np
import sounddevice as sd


def get_uma16_index() -> int:
    """Get the index of the UMA-16 microphone array.

    Searches for UMA-16 by name or by detecting a device with exactly 16 input channels.

    Returns:
        Index of the UMA-16 microphone array.

    Raises RuntimeError if no matching device is found.
    """
    devices = sd.query_devices()
    UMA16_NAMES = ["nanoSHARC micArray16", "UMA16v2", "UMA16", "UMA-16"]

    # Look for UMA-16 by various possible names
    for index, device in enumerate(devices):
        device_name = device["name"]
        for uma_name in UMA16_NAMES:
            if uma_name in device_name:
                return index

    # If not found by name, look for devices with exactly 16 input channels
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) == 16:
            return index

    raise RuntimeError("Could not find the UMA-16 device.")


class SoundDeviceSamplesGeneratorFp64(ac.SoundDeviceSamplesGenerator):
    """Audio sample generator with float64 precision.

    Extends acoular's SoundDeviceSamplesGenerator to convert samples to float64,
    as the sounddevice library's default float32 is not sufficient for the
    precision required by downstream processing.
    """

    def result(self, num):
        self.stream = stream_obj = sd.InputStream(
            device=self.device,
            channels=self.numchannels,
            clip_off=True,
            samplerate=self.sample_freq,
        )
        with stream_obj as stream:
            self.running = True
            if self.numsamples == -1:
                while self.collectsamples:  # yield data as long as collectsamples is True
                    data, self.overflow = stream.read(num)
                    yield data[:num].astype(np.float64)

            elif self.numsamples > 0:  # amount of samples to collect is specified by user
                samples_count = 0  # numsamples counter
                while samples_count < self.numsamples:
                    anz = min(num, self.numsamples - samples_count)
                    data, self.overflow = stream.read(num)
                    yield data[:anz].astype(np.float64)
                    samples_count += anz
        self.running = False
        return

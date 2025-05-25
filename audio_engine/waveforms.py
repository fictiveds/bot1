import numpy
import scipy.signal
from utils.constants import SAMPLE_RATE # Импортируем SAMPLE_RATE

# Примечание: numpy импортируется как numpy, а не np, чтобы соответствовать стилю в sound_generator.py
# scipy.signal импортируется как signal для краткости, если это будет использоваться в других функциях этого модуля.
# Если нет, можно импортировать scipy.signal напрямую при использовании.

def generate_sine_wave(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """
    Генерирует синусоидальную волну (NumPy массив).

    Args:
        frequency (float): Частота синусоиды в Гц.
        duration (float): Длительность синусоиды в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.
        sample_rate (int, optional): Частота дискретизации. Defaults to SAMPLE_RATE.

    Returns:
        numpy.ndarray: Сгенерированная синусоидальная волна.
    """
    time = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * numpy.sin(2 * numpy.pi * frequency * time)
    return wave

def generate_square_wave(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """
    Генерирует прямоугольную волну (square wave) (NumPy массив).

    Args:
        frequency (float): Частота волны в Гц.
        duration (float): Длительность волны в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.
        sample_rate (int, optional): Частота дискретизации. Defaults to SAMPLE_RATE.

    Returns:
        numpy.ndarray: Сгенерированная прямоугольная волна.
    """
    time = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * scipy.signal.square(2 * numpy.pi * frequency * time)
    return wave

def generate_sawtooth_wave(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """
    Генерирует пилообразную волну (NumPy массив).

    Args:
        frequency (float): Частота волны в Гц.
        duration (float): Длительность волны в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.
        sample_rate (int, optional): Частота дискретизации. Defaults to SAMPLE_RATE.

    Returns:
        numpy.ndarray: Сгенерированная пилообразная волна.
    """
    time = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * scipy.signal.sawtooth(2 * numpy.pi * frequency * time)
    return wave

def generate_noise_wave(duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """
    Генерирует случайный шум (NumPy массив).

    Args:
        duration (float): Длительность шума в секундах.
        amplitude (float, optional): Амплитуда шума (от 0.0 до 1.0). Defaults to 0.5.
        sample_rate (int, optional): Частота дискретизации. Defaults to SAMPLE_RATE.

    Returns:
        numpy.ndarray: Сгенерированный шум.
    """
    samples = int(sample_rate * duration) # Количество сэмплов
    wave = numpy.random.rand(samples) * 2 - 1  # Generate samples between -1 and 1
    wave = amplitude * wave
    return wave

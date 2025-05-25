import numpy as np # Используем alias np для единообразия в этом модуле
import scipy.signal
import random

def apply_delay(wave_data, sample_rate, delay_seconds, decay_factor):
    """
    Применяет эффект задержки (эхо) к аудиоданным NumPy.

    Args:
        wave_data (numpy.ndarray): Входной NumPy массив аудиоданных (float).
        sample_rate (int): Частота дискретизации аудио.
        delay_seconds (float): Время задержки в секундах.
        decay_factor (float): Коэффициент затухания для эха (например, 0.5 для половинной амплитуды).

    Returns:
        numpy.ndarray: Обработанный NumPy массив с эффектом задержки.
    """
    delay_samples = int(delay_seconds * sample_rate)  # Количество сэмплов для задержки
    output_len = len(wave_data) + delay_samples
    output_wave = np.zeros(output_len)

    output_wave[:len(wave_data)] = wave_data
    
    # Добавляем задержанную волну
    delayed_part = wave_data * decay_factor
    output_wave[delay_samples:] += delayed_part[:len(output_wave) - delay_samples] # Убедимся, что не пишем за пределы массива

    # Простое клиппирование для удержания значений в диапазоне -1.0 до 1.0
    output_wave = np.clip(output_wave, -1.0, 1.0)
    return output_wave

def apply_filter(wave_data, sample_rate, cutoff_hz, filter_type='lowpass', order=5):
    """
    Применяет фильтр Баттерворта (ФНЧ или ФВЧ) к аудиоданным NumPy.

    Args:
        wave_data (numpy.ndarray): Входной NumPy массив аудиоданных.
        sample_rate (int): Частота дискретизации аудио.
        cutoff_hz (float): Частота среза фильтра в Гц.
        filter_type (str, optional): Тип фильтра: 'lowpass' (ФНЧ) или 'highpass' (ФВЧ). Defaults to 'lowpass'.
        order (int, optional): Порядок фильтра. Defaults to 5.

    Returns:
        numpy.ndarray: Отфильтрованный NumPy массив.
    """
    nyquist = 0.5 * sample_rate  # Частота Найквиста
    normal_cutoff = cutoff_hz / nyquist  # Нормализованная частота среза
    # Создание коэффициентов фильтра Баттерворта
    b, a = scipy.signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_wave = scipy.signal.lfilter(b, a, wave_data)
    return filtered_wave

def apply_reverb(wave_data, sample_rate, number_of_delays=5, max_delay_seconds=0.5, overall_decay_factor=0.6):
    """
    Применяет эффект реверберации к аудиоданным NumPy с использованием нескольких случайных линий задержки.

    Args:
        wave_data (numpy.ndarray): Входной NumPy массив аудиоданных.
        sample_rate (int): Частота дискретизации аудио.
        number_of_delays (int, optional): Количество отдельных линий задержки. Defaults to 5.
        max_delay_seconds (float, optional): Максимальное время задержки для любой линии. Defaults to 0.5.
        overall_decay_factor (float, optional): Общий коэффициент затухания, контролирующий "хвост" реверберации. Defaults to 0.6.

    Returns:
        numpy.ndarray: Обработанный NumPy массив с эффектом реверберации.
    """
    max_delay_samples = int(max_delay_seconds * sample_rate)  # Максимальная задержка в сэмплах
    output_length = len(wave_data) + max_delay_samples
    
    # Начинаем с "сухого" сигнала
    reverb_signal = np.zeros(output_length)
    reverb_signal[:len(wave_data)] = wave_data 

    for _ in range(number_of_delays):
        current_delay_seconds = random.uniform(0.05, max_delay_seconds)
        current_delay_samples = int(current_delay_seconds * sample_rate)
        
        # Затухание для этой конкретной линии задержки, зависящее от общего коэффициента затухания
        current_decay = random.uniform(0.1, 0.5) * overall_decay_factor
        
        echo_source = wave_data * current_decay
        
        # Добавляем компонент задержки
        start_index = current_delay_samples
        
        if start_index < len(reverb_signal): # Убеждаемся, что задержка начинается в пределах буфера вывода
            # Какая часть echo_source может поместиться в reverb_signal
            fit_length = min(len(echo_source), len(reverb_signal) - start_index)
            reverb_signal[start_index : start_index + fit_length] += echo_source[:fit_length]

    # Нормализация, если пиковая амплитуда превышает 1.0, иначе клиппирование
    peak_amplitude = np.max(np.abs(reverb_signal))
    if peak_amplitude > 1.0:
        reverb_signal /= peak_amplitude
    else:
        reverb_signal = np.clip(reverb_signal, -1.0, 1.0)
        
    return reverb_signal

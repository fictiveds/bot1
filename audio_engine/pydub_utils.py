import numpy as np
from pydub import AudioSegment
import random
import soundfile # Для save_wave_to_file

# Относительные импорты NumPy-based функций
from .waveforms import generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise_wave
from .effects import apply_delay, apply_filter, apply_reverb 
# apply_simplified_granular_effect будет здесь, так как он работает с AudioSegment

# Импорт константы
from utils.constants import SAMPLE_RATE

# --- Вспомогательные функции конвертации ---

def _numpy_to_segment(wave_data, sample_rate=SAMPLE_RATE):
    """
    Конвертирует NumPy массив (float от -1.0 до 1.0) в объект pydub AudioSegment.
    Предполагается, что входной массив монофонический или будет сконвертирован в моно.
    """
    if wave_data.ndim == 1: # Моно
        channels = 1
    else: # Potentially stereo, take first channel if more
        channels = 1 # Forcing mono for simplicity with current effects
        wave_data = wave_data[:, 0] if wave_data.shape[1] > 0 else wave_data # take first channel

    # Ensure it's float32 before converting to int16 for consistent behavior
    if wave_data.dtype != np.float32:
        wave_data = wave_data.astype(np.float32)
        
    # Normalize and convert to int16
    int16_wave = (wave_data * 32767).astype(np.int16)
    return AudioSegment(
        int16_wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=int16_wave.dtype.itemsize,
        channels=channels
    )

def _segment_to_numpy(audio_segment):
    """
    Конвертирует объект pydub AudioSegment в NumPy массив (float от -1.0 до 1.0).
    Обрабатывает как моно, так и стерео сегменты (возвращает моно, если стерео).
    """
    samples = np.array(audio_segment.get_array_of_samples())
    
    wave_data = samples / (2**(audio_segment.sample_width * 8 - 1)) 
    
    if audio_segment.channels > 1:
        wave_data = wave_data[::audio_segment.channels] 
    return wave_data.astype(np.float32)

# --- Обёртки для генерации волн (возвращают AudioSegment) ---

def get_sine_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует синусоидальную волну и возвращает ее как AudioSegment."""
    wave_data = generate_sine_wave(frequency, duration, amplitude, sample_rate)
    return _numpy_to_segment(wave_data, sample_rate)

def get_square_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует прямоугольную волну (square wave) и возвращает ее как AudioSegment."""
    wave_data = generate_square_wave(frequency, duration, amplitude, sample_rate)
    return _numpy_to_segment(wave_data, sample_rate)

def get_sawtooth_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует пилообразную волну и возвращает ее как AudioSegment."""
    wave_data = generate_sawtooth_wave(frequency, duration, amplitude, sample_rate)
    return _numpy_to_segment(wave_data, sample_rate)

def get_noise_segment(duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует шумовую волну и возвращает ее как AudioSegment."""
    wave_data = generate_noise_wave(duration, amplitude, sample_rate)
    return _numpy_to_segment(wave_data, sample_rate)

# --- Обёртки для применения эффектов к AudioSegment ---

def apply_delay_to_segment(audio_segment, delay_seconds, decay_factor):
    """Применяет эффект задержки (delay) к AudioSegment и возвращает новый AudioSegment."""
    wave_data = _segment_to_numpy(audio_segment)
    processed_wave = apply_delay(wave_data, audio_segment.frame_rate, delay_seconds, decay_factor)
    return _numpy_to_segment(processed_wave, audio_segment.frame_rate)

def apply_filter_to_segment(audio_segment, cutoff_hz, filter_type='lowpass', order=5):
    """Применяет фильтр к AudioSegment и возвращает новый AudioSegment."""
    wave_data = _segment_to_numpy(audio_segment)
    processed_wave = apply_filter(wave_data, audio_segment.frame_rate, cutoff_hz, filter_type, order)
    return _numpy_to_segment(processed_wave, audio_segment.frame_rate)

def apply_reverb_to_segment(audio_segment, number_of_delays=5, max_delay_seconds=0.5, overall_decay_factor=0.6):
    """Применяет эффект реверберации к AudioSegment и возвращает новый AudioSegment."""
    wave_data = _segment_to_numpy(audio_segment)
    processed_wave = apply_reverb(wave_data, audio_segment.frame_rate, number_of_delays, max_delay_seconds, overall_decay_factor)
    return _numpy_to_segment(processed_wave, audio_segment.frame_rate)

def apply_simplified_granular_effect(audio_segment, grain_duration_ms=50, density=1.0, output_duration_factor=1.0):
    """
    Применяет упрощенный гранулярный эффект к аудиосегменту.

    Args:
        audio_segment (AudioSegment): Входной аудиосегмент.
        grain_duration_ms (int, optional): Длительность каждой гранулы в миллисекундах. Defaults to 50.
        density (float, optional): Фактор, влияющий на количество гранул. 
                                   1.0 означает попытку покрыть исходную длительность. Defaults to 1.0.
        output_duration_factor (float, optional): Множитель для определения длительности выходного звука. 
                                                 1.0 означает примерно ту же длительность. Defaults to 1.0.

    Returns:
        AudioSegment: Обработанный аудиосегмент с гранулярным эффектом.
    """
    if not isinstance(audio_segment, AudioSegment):
        raise TypeError("Input must be an AudioSegment object.")
    if grain_duration_ms <= 0:
        raise ValueError("Grain duration must be positive.")
    if len(audio_segment) == 0:
        return audio_segment 

    if grain_duration_ms >= len(audio_segment):
        return audio_segment[:] 

    num_grains_in_source = int(len(audio_segment) / grain_duration_ms)
    if num_grains_in_source == 0: 
        return audio_segment[:]

    num_grains_to_generate = int(num_grains_in_source * density * output_duration_factor)
    if num_grains_to_generate <= 0:
        return AudioSegment.silent(duration=0, frame_rate=audio_segment.frame_rate)

    output_sound = AudioSegment.empty()
    
    max_start_pos = len(audio_segment) - grain_duration_ms
    if max_start_pos < 0: 
        max_start_pos = 0 

    for _ in range(num_grains_to_generate):
        start_pos = random.randint(0, max_start_pos)
        grain = audio_segment[start_pos : start_pos + grain_duration_ms]
        
        fade_time = min(5, grain_duration_ms // 4)
        if fade_time > 0: 
             grain = grain.fade_in(fade_time).fade_out(fade_time)
        
        output_sound += grain

    if len(output_sound) == 0:
        return audio_segment[:] 
        
    return output_sound

# --- Функции для наложения и смешивания звуков ---

def layer_sounds(segments_to_layer, output_filepath):
    """
    Налаживает (overlay) несколько объектов AudioSegment друг на друга и экспортирует результат в файл.
    Все сегменты начинают воспроизводиться одновременно.

    Args:
        segments_to_layer (list): Список объектов AudioSegment для наложения.
        output_filepath (str): Путь для сохранения смешанного звука (например, "layered_output.wav").
    """
    if not segments_to_layer:
        print("Предупреждение: Список сегментов для наложения пуст.")
        return

    mixed_sound = segments_to_layer[0]
    for i in range(1, len(segments_to_layer)):
        mixed_sound = mixed_sound.overlay(segments_to_layer[i])
    
    try:
        mixed_sound.export(output_filepath, format="wav")
        print(f"Layered sound saved to: {output_filepath}")
    except Exception as e:
        print(f"Error exporting layered sound: {e}")


def mix_sounds(sound_list_with_positions, output_filepath, main_track_duration_ms=None):
    """
    Смешивает несколько объектов AudioSegment в указанных временных позициях и экспортирует результат в файл.

    Args:
        sound_list_with_positions (list): Список словарей, где каждый словарь имеет вид
                                          {'segment': AudioSegment, 'start_time_ms': int (время начала в мс)}.
        output_filepath (str): Путь для сохранения смешанного звука (например, "mixed_output.wav").
        main_track_duration_ms (int, optional): Общая длительность основного трека в миллисекундах. 
                                                Если None, вычисляется по времени окончания последнего сегмента.
    """
    if not sound_list_with_positions:
        print("Предупреждение: Список сегментов для смешивания пуст.")
        return

    if main_track_duration_ms is None:
        main_track_duration_ms = 0
        for item in sound_list_with_positions:
            segment_end_time = item['start_time_ms'] + len(item['segment'])
            if segment_end_time > main_track_duration_ms:
                main_track_duration_ms = segment_end_time
    
    if main_track_duration_ms == 0:
        print("Warning: All segments have zero length or no start times defined; output will be empty.")
        AudioSegment.silent(duration=1, frame_rate=SAMPLE_RATE).export(output_filepath, format="wav")
        return

    main_track = AudioSegment.silent(duration=main_track_duration_ms, frame_rate=SAMPLE_RATE)

    for item in sound_list_with_positions:
        main_track = main_track.overlay(item['segment'], position=item['start_time_ms'])
            
    try:
        main_track.export(output_filepath, format="wav")
        print(f"Mixed sound saved to: {output_filepath}")
    except Exception as e:
        print(f"Error exporting mixed sound: {e}")

# --- Функция сохранения NumPy-массива (для отладки или специфических нужд) ---
def save_wave_to_file(filepath, wave_data, sample_rate=SAMPLE_RATE):
    """
    Сохраняет аудиоданные (NumPy массив) в WAV-файл.

    Args:
        filepath (str): Путь для сохранения WAV-файла.
        wave_data (numpy.ndarray): Аудиоданные для сохранения.
        sample_rate (int, optional): Частота дискретизации. Defaults to SAMPLE_RATE.
    """
    soundfile.write(filepath, wave_data, sample_rate)

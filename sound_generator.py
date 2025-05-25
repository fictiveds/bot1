import numpy
import soundfile
from scipy import signal
import random
import time # Для генерации случайного seed по умолчанию
import argparse
from pydub import AudioSegment
import numpy as np # Already imported as numpy, but requirement asks for np specifically

SAMPLE_RATE = 44100  # Глобальная частота дискретизации для всего проекта (сэмплов в секунду)


def get_random_params(max_duration_seconds=4.0):
    """
    Генерирует словарь со случайными параметрами для генерации звука.

    Args:
        max_duration_seconds (float, optional): Максимальная длительность генерируемого звука в секундах. 
                                                Defaults to 4.0.

    Returns:
        dict: Словарь, содержащий:
            'waveform_type' (str): Случайно выбранный из ['sine', 'square', 'sawtooth', 'noise'].
            'frequency' (float): Случайное число от 50 Гц до 2000 Гц (если применимо).
            'duration' (float): Случайное число от 0.2 до `max_duration_seconds`.
            'amplitude' (float): Случайное число от 0.1 до 0.7.
    """
    waveform_type = random.choice(['sine', 'square', 'sawtooth', 'noise'])
    
    # Ensure min_duration is less than max_duration_seconds, adjust if necessary
    min_duration = 0.2
    if min_duration >= max_duration_seconds:
        min_duration = max_duration_seconds / 2 if max_duration_seconds > 0 else 0.1

    params = {
        'waveform_type': waveform_type,
        'duration': random.uniform(min_duration, max_duration_seconds),
        'amplitude': random.uniform(0.1, 0.7), # Slightly reduced max amplitude for mixing
    }
    if waveform_type != 'noise':
        params['frequency'] = random.uniform(50.0, 2000.0)
    return params


def create_experimental_composition(composition_duration_seconds=30, num_layers=5, events_per_layer_range=(1,5), sample_rate=SAMPLE_RATE):
    """
    Создает экспериментальную звуковую композицию путем наложения множества случайно сгенерированных
    и обработанных звуковых событий.

    Args:
        composition_duration_seconds (int, optional): Желаемая общая длительность композиции в секундах. Defaults to 30.
        num_layers (int, optional): Количество независимых звуковых "слоев" или "дорожек". Defaults to 5.
        events_per_layer_range (tuple, optional): Диапазон (min, max) случайного количества событий на слой. Defaults to (1,5).
        sample_rate (int, optional): Частота дискретизации для генерации. Defaults to SAMPLE_RATE.

    Returns:
        AudioSegment: Финальная смешанная аудиокомпозиция.
    """
    composition_duration_ms = composition_duration_seconds * 1000
    sound_events = []

    available_effects = [
        'delay', 'filter_lowpass', 'filter_highpass', 'reverb', 'granular'
    ]

    for _ in range(num_layers):
        num_events_this_layer = random.randint(events_per_layer_range[0], events_per_layer_range[1])
        
        for _ in range(num_events_this_layer):
            # Max duration for individual events, e.g., 1/5th of total duration or a fixed cap like 8s
            max_event_duration = min(8.0, composition_duration_seconds / 3.0) 
            event_params = get_random_params(max_duration_seconds=max_event_duration)
            
            base_segment = None
            if event_params['waveform_type'] == 'sine':
                base_segment = get_sine_segment(event_params.get('frequency', 440), event_params['duration'], event_params['amplitude'], sample_rate)
            elif event_params['waveform_type'] == 'square':
                base_segment = get_square_segment(event_params.get('frequency', 440), event_params['duration'], event_params['amplitude'], sample_rate)
            elif event_params['waveform_type'] == 'sawtooth':
                base_segment = get_sawtooth_segment(event_params.get('frequency', 440), event_params['duration'], event_params['amplitude'], sample_rate)
            elif event_params['waveform_type'] == 'noise':
                base_segment = get_noise_segment(event_params['duration'], event_params['amplitude'], sample_rate)

            if not base_segment or len(base_segment) == 0:
                continue

            processed_segment = base_segment

            # Apply 0 to 3 effects randomly, with a bias towards more effects
            # num_effects_to_apply = random.randint(0, 3) # Old way
            num_effects_to_apply = random.choice([0, 1, 1, 2, 2, 2, 3, 3]) # Bias towards 1, 2 or 3 effects
            chosen_effects = random.sample(available_effects, k=min(num_effects_to_apply, len(available_effects)))
            
            # print(f"    Выбрано эффектов: {num_effects_to_apply}, список: {chosen_effects}") # Отладочный принт
            for effect_name in chosen_effects:
                if len(processed_segment) == 0: break # Skip if segment became empty

                if effect_name == 'delay':
                    processed_segment = apply_delay_to_segment(processed_segment, 
                                                               delay_seconds=random.uniform(0.05, 0.4), 
                                                               decay_factor=random.uniform(0.2, 0.6))
                elif effect_name == 'filter_lowpass':
                    cutoff = random.uniform(200, 3000)
                    if processed_segment.frame_rate / 2 > cutoff + 100: # Ensure cutoff is below Nyquist
                         processed_segment = apply_filter_to_segment(processed_segment, cutoff_hz=cutoff, filter_type='lowpass')
                elif effect_name == 'filter_highpass':
                    cutoff = random.uniform(200, 3000)
                    if processed_segment.frame_rate / 2 > cutoff + 100: # Ensure cutoff is below Nyquist
                        processed_segment = apply_filter_to_segment(processed_segment, cutoff_hz=cutoff, filter_type='highpass')
                elif effect_name == 'reverb':
                    processed_segment = apply_reverb_to_segment(processed_segment,
                                                                number_of_delays=random.randint(3, 7),
                                                                max_delay_seconds=random.uniform(0.1, 0.5),
                                                                overall_decay_factor=random.uniform(0.2, 0.5))
                elif effect_name == 'granular':
                    if len(processed_segment) > 20: # Granular effect needs some length
                        processed_segment = apply_simplified_granular_effect(processed_segment,
                                                                         grain_duration_ms=random.randint(20, 100),
                                                                         density=random.uniform(0.5, 1.5),
                                                                         output_duration_factor=random.uniform(0.8, 1.2))
            
            if len(processed_segment) > 0:
                event_start_ms = random.randint(0, max(0, composition_duration_ms - len(processed_segment)))
                # Reduce volume slightly for mixing, more for background layers
                processed_segment = processed_segment - random.uniform(3, 9) # Reduce dB

                # Случайное панорамирование
                # pydub панорамирует от -1.0 (полностью влево) до 1.0 (полностью вправо)
                random_pan = random.uniform(-0.8, 0.8) # Диапазон не до упора, чтобы не терять звук полностью
                processed_segment = processed_segment.pan(random_pan)
                # print(f"    Применено панорамирование: {random_pan:.2f} к событию.") # Отладочный принт
                
                sound_events.append({'segment': processed_segment, 'start_time_ms': event_start_ms})

    if not sound_events:
        # Fallback: create a single simple sound if nothing was generated
        fallback_params = get_random_params(max_duration_seconds=2.0)
        fallback_segment = get_sine_segment(fallback_params.get('frequency', 220), fallback_params['duration'], fallback_params['amplitude'])
        # Применяем панорамирование и к запасному звуку
        random_pan_fallback = random.uniform(-0.5, 0.5)
        fallback_segment = (fallback_segment - 6).pan(random_pan_fallback)
        sound_events.append({'segment': fallback_segment, 'start_time_ms': 0})

    # Инициализация final_composition как стерео
    final_composition = AudioSegment.silent(duration=composition_duration_ms, frame_rate=sample_rate).set_channels(2)
    
    for event in sound_events:
        # Убедимся, что накладываемый сегмент тоже стерео, если он моно.
        # .pan() уже должен делать сегмент стерео.
        # Если final_composition стерео, а event.segment моно, pydub смешает моно в оба канала.
        final_composition = final_composition.overlay(event['segment'], position=event['start_time_ms'])
    
    final_composition = final_composition.normalize()
    
    return final_composition


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
    
    # Add delayed wave
    delayed_part = wave_data * decay_factor
    output_wave[delay_samples:] += delayed_part[:len(output_wave) - delay_samples] # Ensure not to write past output_wave

    # Simple clipping to keep values within -1.0 to 1.0
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
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_wave = signal.lfilter(b, a, wave_data)
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
    
    # Start with the dry signal
    reverb_signal = np.zeros(output_length)
    reverb_signal[:len(wave_data)] = wave_data 

    for _ in range(number_of_delays):
        current_delay_seconds = random.uniform(0.05, max_delay_seconds)
        current_delay_samples = int(current_delay_seconds * sample_rate)
        
        # Decay for this specific delay line, influenced by overall_decay_factor
        current_decay = random.uniform(0.1, 0.5) * overall_decay_factor
        
        echo_source = wave_data * current_decay
        
        # Add the delayed component
        start_index = current_delay_samples
        end_index = start_index + len(echo_source)
        
        if start_index < len(reverb_signal): # Ensure the delay starts within the output buffer
            # How much of the echo_source can fit into reverb_signal
            fit_length = min(len(echo_source), len(reverb_signal) - start_index)
            reverb_signal[start_index : start_index + fit_length] += echo_source[:fit_length]

    # Normalize if peak amplitude exceeds 1.0, otherwise clip
    peak_amplitude = np.max(np.abs(reverb_signal))
    if peak_amplitude > 1.0:
        reverb_signal /= peak_amplitude
    else:
        reverb_signal = np.clip(reverb_signal, -1.0, 1.0)
        
    return reverb_signal

# --- AudioSegment Wrapper Functions ---

def _numpy_to_segment(wave_data, sample_rate):
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
    
    # Нормализация до диапазона float -1.0 до 1.0
    # audio_segment.sample_width это ширина сэмпла в байтах (например, 2 для 16-бит)
    wave_data = samples / (2**(audio_segment.sample_width * 8 - 1)) 
    
    # Если сегмент был стерео, pydub.get_array_of_samples() чередует каналы.
    # Для простоты дальнейшей обработки эффектами, которые ожидают моно,
    # можно взять один канал или усреднить. Здесь берется первый канал.
    if audio_segment.channels > 1:
        # Предполагается, что NumPy-эффекты ожидают моно или должны быть адаптированы
        # Это упрощение может потребовать пересмотра для стерео-эффектов
        wave_data = wave_data[::audio_segment.channels] # Берем сэмплы первого канала
    # Ensure it's float32 for processing by existing numpy functions
    return wave_data.astype(np.float32)

# --- Sound Generation Wrappers (Return AudioSegment) ---

def get_sine_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует синусоидальную волну и возвращает ее как AudioSegment."""
    wave_data = generate_sine_wave(frequency, duration, amplitude)
    return _numpy_to_segment(wave_data, sample_rate)

def get_square_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует прямоугольную волну (square wave) и возвращает ее как AudioSegment."""
    wave_data = generate_square_wave(frequency, duration, amplitude)
    return _numpy_to_segment(wave_data, sample_rate)

def get_sawtooth_segment(frequency, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует пилообразную волну и возвращает ее как AudioSegment."""
    wave_data = generate_sawtooth_wave(frequency, duration, amplitude)
    return _numpy_to_segment(wave_data, sample_rate)

def get_noise_segment(duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Генерирует шумовую волну и возвращает ее как AudioSegment."""
    wave_data = generate_noise_wave(duration, amplitude)
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
        # Create a very short silent file if needed, or handle as error
        AudioSegment.silent(duration=1).export(output_filepath, format="wav")
        return

    main_track = AudioSegment.silent(duration=main_track_duration_ms, frame_rate=SAMPLE_RATE)

    for item in sound_list_with_positions:
        main_track = main_track.overlay(item['segment'], position=item['start_time_ms'])
            
    try:
        main_track.export(output_filepath, format="wav")
        print(f"Mixed sound saved to: {output_filepath}")
    except Exception as e:
        print(f"Error exporting mixed sound: {e}")


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
        return audio_segment # Return empty segment if input is empty

    # Если grain_duration_ms больше длины сегмента, вернуть часть или копию.
    if grain_duration_ms >= len(audio_segment):
        # Можно вернуть просто копию, или первую часть длительностью grain_duration_ms, если density=1 и factor=1
        # Для упрощения вернем копию, так как эффект не сможет "гранулировать"
        return audio_segment[:] 

    num_grains_in_source = int(len(audio_segment) / grain_duration_ms)
    if num_grains_in_source == 0: # Should be caught by above check, but as safety
        return audio_segment[:]

    num_grains_to_generate = int(num_grains_in_source * density * output_duration_factor)
    if num_grains_to_generate <= 0:
        return AudioSegment.silent(duration=0, frame_rate=audio_segment.frame_rate)

    output_sound = AudioSegment.empty()
    
    max_start_pos = len(audio_segment) - grain_duration_ms
    if max_start_pos < 0: # Should ideally not happen if grain_duration_ms check above is correct
        max_start_pos = 0 

    for _ in range(num_grains_to_generate):
        start_pos = random.randint(0, max_start_pos)
        grain = audio_segment[start_pos : start_pos + grain_duration_ms]
        
        # Применение короткого fade in/out
        fade_time = min(5, grain_duration_ms // 4)
        if fade_time > 0: # fade_in/out might fail on zero duration fades for very short grains
             grain = grain.fade_in(fade_time).fade_out(fade_time)
        
        output_sound += grain

    if len(output_sound) == 0:
        # Это может произойти, если, например, num_grains_to_generate было 0.
        return audio_segment[:] # Возвращаем копию исходного сегмента
        
    return output_sound


def generate_sine_wave(frequency, duration, amplitude=0.5):
    """
    Генерирует синусоидальную волну (NumPy массив).

    Args:
        frequency (float): Частота синусоиды в Гц.
        duration (float): Длительность синусоиды в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.

    Returns:
        numpy.ndarray: Сгенерированная синусоидальная волна.
    """
    time = numpy.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False) # endpoint=False для точного количества сэмплов
    wave = amplitude * numpy.sin(2 * numpy.pi * frequency * time)
    return wave

def generate_square_wave(frequency, duration, amplitude=0.5):
    """
    Генерирует прямоугольную волну (square wave) (NumPy массив).

    Args:
        frequency (float): Частота волны в Гц.
        duration (float): Длительность волны в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.

    Returns:
        numpy.ndarray: Сгенерированная прямоугольная волна.
    """
    time = numpy.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = amplitude * signal.square(2 * numpy.pi * frequency * time)
    return wave

def generate_sawtooth_wave(frequency, duration, amplitude=0.5):
    """
    Генерирует пилообразную волну (NumPy массив).

    Args:
        frequency (float): Частота волны в Гц.
        duration (float): Длительность волны в секундах.
        amplitude (float, optional): Амплитуда волны (от 0.0 до 1.0). Defaults to 0.5.

    Returns:
        numpy.ndarray: Сгенерированная пилообразная волна.
    """
    time = numpy.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = amplitude * signal.sawtooth(2 * numpy.pi * frequency * time)
    return wave

def generate_noise_wave(duration, amplitude=0.5):
    """
    Генерирует случайный шум (NumPy массив).

    Args:
        duration (float): Длительность шума в секундах.
        amplitude (float, optional): Амплитуда шума (от 0.0 до 1.0). Defaults to 0.5.

    Returns:
        numpy.ndarray: Сгенерированный шум.
    """
    samples = int(SAMPLE_RATE * duration) # Количество сэмплов
    wave = numpy.random.rand(samples) * 2 - 1  # Generate samples between -1 and 1
    wave = amplitude * wave
    return wave

def save_wave_to_file(filepath, wave_data, sample_rate):
    """
    Сохраняет аудиоданные (NumPy массив) в WAV-файл.

    Args:
        filepath (str): Путь для сохранения WAV-файла.
        wave_data (numpy.ndarray): Аудиоданные для сохранения.
        sample_rate (int): Частота дискретизации.
    """
    soundfile.write(filepath, wave_data, sample_rate)

if __name__ == "__main__":
    # --- Начало блока CLI ---
    # Старый демонстрационный код закомментирован или удален, 
    # так как теперь управление идет через аргументы командной строки.

    # Все строки ниже, которые были частью старого демонстрационного кода, будут удалены.
    # Они уже закомментированы, но задача требует их полного удаления.

    parser = argparse.ArgumentParser(description="Генератор экспериментальной музыки и случайных звуков.")
    parser.add_argument('--output_file', '-o', type=str, default='experimental_music.wav',
                        help='Имя выходного WAV-файла для композиции.')
    parser.add_argument('--duration', '-d', type=int, default=30, 
                        help='Длительность композиции в секундах.')
    parser.add_argument('--layers', '-l', type=int, default=5, 
                        help='Количество слоев в композиции.')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Зерно для генератора случайных чисел. Если не указано, используется случайное.')
    parser.add_argument('--single_sound_file', '-s', type=str, default=None,
                        help='(Опционально) Если указано, генерирует один случайный звук и сохраняет его в этот файл, вместо полной композиции.')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed) # Также для numpy, если он используется для случайности напрямую
        print(f"Использовано зерно для случайности: {args.seed}")
    else:
        # Используем текущее время как seed по умолчанию для большей случайности при каждом запуске
        current_seed = int(time.time())
        random.seed(current_seed)
        np.random.seed(current_seed)
        print(f"Использовано случайное зерно (на основе времени): {current_seed}")

    if args.single_sound_file:
        print(f"Генерация одного случайного звука в файл: {args.single_sound_file}")
        # Генерируем один случайный звук
        # Max duration for a single sound, can be different from composition event duration
        single_sound_params = get_random_params(max_duration_seconds=random.uniform(1.0, 5.0)) 
        
        base_segment = None
        if single_sound_params['waveform_type'] == 'sine':
            base_segment = get_sine_segment(single_sound_params.get('frequency', 440), single_sound_params['duration'], single_sound_params['amplitude'])
        elif single_sound_params['waveform_type'] == 'square':
            base_segment = get_square_segment(single_sound_params.get('frequency', 440), single_sound_params['duration'], single_sound_params['amplitude'])
        elif single_sound_params['waveform_type'] == 'sawtooth':
            base_segment = get_sawtooth_segment(single_sound_params.get('frequency', 440), single_sound_params['duration'], single_sound_params['amplitude'])
        elif single_sound_params['waveform_type'] == 'noise':
            base_segment = get_noise_segment(single_sound_params['duration'], single_sound_params['amplitude'])

        if base_segment and len(base_segment) > 0:
            processed_segment = base_segment
            
            # Применить 0-2 случайных эффекта
            available_effects = ['delay', 'filter_lowpass', 'filter_highpass', 'reverb', 'granular']
            num_effects_to_apply = random.randint(0, 2)
            chosen_effects = random.sample(available_effects, k=min(num_effects_to_apply, len(available_effects)))

            for effect_name in chosen_effects:
                if len(processed_segment) == 0: break
                if effect_name == 'delay':
                    processed_segment = apply_delay_to_segment(processed_segment, random.uniform(0.1, 0.5), random.uniform(0.2, 0.6))
                elif effect_name == 'filter_lowpass':
                     cutoff = random.uniform(200, 3000)
                     if processed_segment.frame_rate / 2 > cutoff + 100:
                        processed_segment = apply_filter_to_segment(processed_segment, cutoff, 'lowpass')
                elif effect_name == 'filter_highpass':
                    cutoff = random.uniform(200, 3000)
                    if processed_segment.frame_rate / 2 > cutoff + 100:
                        processed_segment = apply_filter_to_segment(processed_segment, cutoff, 'highpass')
                elif effect_name == 'reverb':
                    processed_segment = apply_reverb_to_segment(processed_segment, random.randint(3,7), random.uniform(0.1,0.6), random.uniform(0.2,0.5))
                elif effect_name == 'granular':
                    if len(processed_segment) > 20:
                         processed_segment = apply_simplified_granular_effect(processed_segment, random.randint(20,100), random.uniform(0.5,1.5), random.uniform(0.8,1.2))
            
            try:
                processed_segment.export(args.single_sound_file, format="wav")
                print(f"Случайный звук сохранен в: {args.single_sound_file}")
            except Exception as e:
                print(f"Ошибка при сохранении случайного звука: {e}")
        else:
            print("Не удалось сгенерировать базовый сегмент для случайного звука.")

    else:
        print(f"Генерация экспериментальной композиции в файл: {args.output_file}")
        experimental_comp = create_experimental_composition(
            composition_duration_seconds=args.duration, 
            num_layers=args.layers
        )
        if experimental_comp:
            try:
                experimental_comp.export(args.output_file, format="wav")
                print(f"Экспериментальная композиция сохранена в: {args.output_file}")
            except Exception as e:
                print(f"Ошибка при сохранении композиции: {e}")
        else:
            print("Не удалось создать экспериментальную композицию.")

# --- Конец блока CLI ---
# Весь старый демонстрационный код, который ранее находился здесь (закомментированный), был удален.
# Остался только код, относящийся к интерфейсу командной строки.

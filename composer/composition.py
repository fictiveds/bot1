import random
import numpy as np # Хотя np напрямую не используется, он может быть частью зависимостей pydub или других эффектов
from pydub import AudioSegment

# Импорт функций для генерации и обработки сегментов
from audio_engine.pydub_utils import (
    get_sine_segment, get_square_segment, get_sawtooth_segment, get_noise_segment,
    apply_delay_to_segment, apply_filter_to_segment, apply_reverb_to_segment,
    apply_simplified_granular_effect
)
# Импорт константы
from utils.constants import SAMPLE_RATE

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
    
    min_duration = 0.2
    if min_duration >= max_duration_seconds:
        min_duration = max_duration_seconds / 2 if max_duration_seconds > 0 else 0.1

    params = {
        'waveform_type': waveform_type,
        'duration': random.uniform(min_duration, max_duration_seconds),
        'amplitude': random.uniform(0.1, 0.7), 
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
            max_event_duration = min(8.0, composition_duration_seconds / 3.0) 
            event_params = get_random_params(max_duration_seconds=max_event_duration)
            
            base_segment = None
            common_args = {
                'duration': event_params['duration'], 
                'amplitude': event_params['amplitude'], 
                'sample_rate': sample_rate
            }
            freq_arg = event_params.get('frequency', 440)

            if event_params['waveform_type'] == 'sine':
                base_segment = get_sine_segment(frequency=freq_arg, **common_args)
            elif event_params['waveform_type'] == 'square':
                base_segment = get_square_segment(frequency=freq_arg, **common_args)
            elif event_params['waveform_type'] == 'sawtooth':
                base_segment = get_sawtooth_segment(frequency=freq_arg, **common_args)
            elif event_params['waveform_type'] == 'noise':
                base_segment = get_noise_segment(duration=event_params['duration'], amplitude=event_params['amplitude'], sample_rate=sample_rate)

            if not base_segment or len(base_segment) == 0:
                continue

            processed_segment = base_segment
            num_effects_to_apply = random.choice([0, 1, 1, 2, 2, 2, 3, 3])
            chosen_effects = random.sample(available_effects, k=min(num_effects_to_apply, len(available_effects)))
            
            for effect_name in chosen_effects:
                if len(processed_segment) == 0: break 

                if effect_name == 'delay':
                    processed_segment = apply_delay_to_segment(processed_segment, 
                                                               delay_seconds=random.uniform(0.05, 0.4), 
                                                               decay_factor=random.uniform(0.2, 0.6))
                elif effect_name == 'filter_lowpass':
                    cutoff = random.uniform(200, 3000)
                    if processed_segment.frame_rate / 2 > cutoff + 100: 
                         processed_segment = apply_filter_to_segment(processed_segment, cutoff_hz=cutoff, filter_type='lowpass')
                elif effect_name == 'filter_highpass':
                    cutoff = random.uniform(200, 3000)
                    if processed_segment.frame_rate / 2 > cutoff + 100: 
                        processed_segment = apply_filter_to_segment(processed_segment, cutoff_hz=cutoff, filter_type='highpass')
                elif effect_name == 'reverb':
                    processed_segment = apply_reverb_to_segment(processed_segment,
                                                                number_of_delays=random.randint(3, 7),
                                                                max_delay_seconds=random.uniform(0.1, 0.5),
                                                                overall_decay_factor=random.uniform(0.2, 0.5))
                elif effect_name == 'granular':
                    if len(processed_segment) > 20: 
                        processed_segment = apply_simplified_granular_effect(processed_segment,
                                                                         grain_duration_ms=random.randint(20, 100),
                                                                         density=random.uniform(0.5, 1.5),
                                                                         output_duration_factor=random.uniform(0.8, 1.2))
            
            if len(processed_segment) > 0:
                event_start_ms = random.randint(0, max(0, composition_duration_ms - len(processed_segment)))
                processed_segment = processed_segment - random.uniform(3, 9) 
                random_pan = random.uniform(-0.8, 0.8) 
                processed_segment = processed_segment.pan(random_pan)
                sound_events.append({'segment': processed_segment, 'start_time_ms': event_start_ms})

    if not sound_events:
        fallback_params = get_random_params(max_duration_seconds=2.0)
        fallback_segment = get_sine_segment(
            frequency=fallback_params.get('frequency', 220), 
            duration=fallback_params['duration'], 
            amplitude=fallback_params['amplitude'],
            sample_rate=sample_rate
        )
        random_pan_fallback = random.uniform(-0.5, 0.5)
        fallback_segment = (fallback_segment - 6).pan(random_pan_fallback)
        sound_events.append({'segment': fallback_segment, 'start_time_ms': 0})

    final_composition = AudioSegment.silent(duration=composition_duration_ms, frame_rate=sample_rate).set_channels(2)
    
    for event in sound_events:
        final_composition = final_composition.overlay(event['segment'], position=event['start_time_ms'])
    
    final_composition = final_composition.normalize()
    
    return final_composition

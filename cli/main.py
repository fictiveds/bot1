import argparse
import random
import time
import numpy as np
from pydub import AudioSegment # Хотя бы для type hinting, если понадобится

# Импорты из созданных модулей
from composer.composition import create_experimental_composition, get_random_params
from audio_engine.pydub_utils import (
    get_sine_segment, get_square_segment, get_sawtooth_segment, get_noise_segment,
    apply_delay_to_segment, apply_filter_to_segment, apply_reverb_to_segment,
    apply_simplified_granular_effect
)
from utils.constants import SAMPLE_RATE # Используется в функциях, но хорошо иметь доступ при необходимости

def main():
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

    current_seed = args.seed
    if current_seed is None:
        current_seed = int(time.time())
        print(f"Использовано случайное зерно (на основе времени): {current_seed}")
    else:
        print(f"Использовано зерно для случайности: {current_seed}")
    
    random.seed(current_seed)
    np.random.seed(current_seed)

    if args.single_sound_file:
        print(f"Генерация одного случайного звука в файл: {args.single_sound_file}")
        
        single_sound_params = get_random_params(max_duration_seconds=random.uniform(1.0, 5.0)) 
        
        base_segment = None
        common_args_single = {
            'duration': single_sound_params['duration'], 
            'amplitude': single_sound_params['amplitude'],
            'sample_rate': SAMPLE_RATE # Явно передаем SAMPLE_RATE
        }
        freq_arg_single = single_sound_params.get('frequency', 440)

        if single_sound_params['waveform_type'] == 'sine':
            base_segment = get_sine_segment(frequency=freq_arg_single, **common_args_single)
        elif single_sound_params['waveform_type'] == 'square':
            base_segment = get_square_segment(frequency=freq_arg_single, **common_args_single)
        elif single_sound_params['waveform_type'] == 'sawtooth':
            base_segment = get_sawtooth_segment(frequency=freq_arg_single, **common_args_single)
        elif single_sound_params['waveform_type'] == 'noise':
            base_segment = get_noise_segment(duration=single_sound_params['duration'], amplitude=single_sound_params['amplitude'], sample_rate=SAMPLE_RATE)

        if base_segment and len(base_segment) > 0:
            processed_segment = base_segment
            
            available_effects = ['delay', 'filter_lowpass', 'filter_highpass', 'reverb', 'granular']
            num_effects_to_apply = random.randint(0, 2) # Для одиночного звука оставим 0-2 эффекта
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
                # Убедимся, что сегмент стерео перед экспортом, если он моно
                if processed_segment.channels == 1:
                    processed_segment = processed_segment.set_channels(2)
                processed_segment.export(args.single_sound_file, format="wav")
                print(f"Случайный звук сохранен в: {args.single_sound_file}")
            except Exception as e:
                print(f"Ошибка при сохранении случайного звука: {e}")
        else:
            print("Не удалось сгенерировать базовый сегмент для случайного звука.")

    else:
        print(f"Генерация экспериментальной композиции в файл: {args.output_file}")
        # SAMPLE_RATE передается в create_experimental_composition по умолчанию из ее определения
        experimental_comp = create_experimental_composition(
            composition_duration_seconds=args.duration, 
            num_layers=args.layers,
            sample_rate=SAMPLE_RATE # Явно передаем для ясности
        )
        if experimental_comp:
            try:
                # .set_channels(2) уже должен быть вызван в create_experimental_composition
                experimental_comp.export(args.output_file, format="wav")
                print(f"Экспериментальная композиция сохранена в: {args.output_file}")
            except Exception as e:
                print(f"Ошибка при сохранении композиции: {e}")
        else:
            print("Не удалось создать экспериментальную композицию.")

if __name__ == "__main__":
    main()

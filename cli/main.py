import argparse
import random
import time
import numpy as np
from pydub import AudioSegment # Хотя бы для type hinting, если понадобится

# Импорты из созданных модулей
from composer.composition import create_experimental_composition, get_random_params
from audio_engine.pydub_utils import ( # noqa
    get_sine_segment, get_square_segment, get_sawtooth_segment, get_noise_segment, # noqa
    apply_delay_to_segment, apply_filter_to_segment, apply_reverb_to_segment, # noqa
    apply_simplified_granular_effect # noqa
)
from utils.constants import SAMPLE_RATE
from nodal_engine.graph import SoundGraph
from nodal_engine.modules import SineOscillator, LFO


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
    parser.add_argument(
            '--nodal_example',
            type=str,
            metavar='FILENAME.WAV', # Чтобы пользователь указывал имя файла
            help='Сгенерировать простой пример звука с использованием нодального движка и сохранить в указанный FILENAME.WAV. '
                 'Это переопределяет другие режимы генерации (композиция/одиночный звук).'
        )
    args = parser.parse_args()

    current_seed = args.seed
    if current_seed is None:
        current_seed = int(time.time())
        print(f"Использовано случайное зерно (на основе времени): {current_seed}")
    else:
        print(f"Использовано зерно для случайности: {current_seed}")
    
    random.seed(current_seed)
    np.random.seed(current_seed)

    if args.nodal_example:
        print(f"Генерация примера нодального звука в файл: {args.nodal_example}")
        try:
            # 1. Создаем граф
            graph = SoundGraph()

            # 2. Создаем модули
            # LFO будет модулировать амплитуду синусоиды.
            # LFO генерирует сигнал в диапазоне [-amplitude, +amplitude].
            # Для модуляции амплитуды синусоиды (которая обычно 0.0-1.0) нам нужно,
            # чтобы LFO выдавал значения, например, от 0.0 до 1.0 (или 0.0 до 0.5 для меньшей громкости).
            # Если LFO.amplitude = 0.5, он дает [-0.5, +0.5]. Если прибавить 0.5, получим [0.0, 1.0].
            # Это можно сделать, если LFO имеет параметр 'offset' или если есть модуль Adder.
            # Пока что LFO будет модулировать амплитуду "как есть", что может привести к инверсии фазы.
            # Для более контролируемого эффекта тремоло, выход LFO должен быть смещен и масштабирован.
            
            # Пример: LFO для амплитуды, создающий тремоло.
            # LFO генерирует значения от -0.4 до +0.4.
            # Если мы хотим, чтобы амплитуда основного звука менялась, например, от 0.1 до 0.9,
            # то (LFO_output + 1) * 0.4 + 0.1 -> ([-0.4, 0.4] + 1) * 0.4 + 0.1 -> [0.6, 1.4]*0.4 +0.1 -> [0.24, 0.56] + 0.1 -> [0.34, 0.66]
            # Это потребует модулей смещения и масштабирования.
            # В данном примере мы просто подключим LFO напрямую, что приведет к модуляции амплитуды
            # синусоиды значениями LFO. Если амплитуда LFO 0.5, то выход синусоиды будет исходная_волна * [-0.5, 0.5].
            # Это не совсем то, что нужно для простого тремоло, но демонстрирует подключение.
            # Чтобы сделать эффект тремоло, где амплитуда меняется, скажем, от 0 (тишина) до 0.5 (макс. громкость LFO),
            # LFO должен генерировать сигнал в диапазоне [0, 1] (или [0, X]), а затем этот сигнал умножается на желаемую амплитуду.
            # (LFO_sine(-1..1) + 1) / 2 -> [0..1]
            # Для этого LFO должен быть способен настраивать свой выходной диапазон или нужен модуль Scale/Offset.
            # Пока оставим как есть: LFO модулирует амплитуду синусоиды.
            
            lfo_for_amp = LFO(name="amp_lfo", frequency=2.0, amplitude=0.5) # Генерирует значения от -0.5 до +0.5
            # Базовая амплитуда синусоиды будет 0.5.
            # Выход LFO будет использоваться для модуляции этой базовой амплитуды.
            # Фактически, значение с LFO (от -0.5 до +0.5) будет само по себе новой амплитудой.
            # Это не совсем модуляция, а прямое управление амплитудой через LFO.
            # Для истинной амплитудной модуляции (AM), нужно: out = carrier * (1 + modulator) или carrier * modulator
            # В нашем случае, get_input_value для 'amplitude' просто заменит self._amplitude.
            
            sine_osc = SineOscillator(name="sine1", frequency=220.0, amplitude=0.0) # Начальная амплитуда 0, будет полностью управляться LFO

            # 3. Добавляем модули в граф
            graph.add_module(lfo_for_amp)
            graph.add_module(sine_osc)

            # 4. Соединяем: выход 'value' LFO к входу 'amplitude' осциллятора
            graph.connect(source_module_name="amp_lfo", source_output_name="value",
                          target_module_name="sine1", target_input_name="amplitude")

            # 5. Устанавливаем мастер-выход
            graph.set_master_output("sine1")

            # 6. Рендерим аудио
            duration = 5.0  # секунд
            output_audio_segment = graph.render_audio(duration, SAMPLE_RATE)

            # 7. Сохраняем результат
            output_audio_segment.export(args.nodal_example, format="wav")
            print(f"Нодальный пример сохранен в: {args.nodal_example}")

        except Exception as e:
            print(f"Ошибка при генерации нодального примера: {e}")
            import traceback
            traceback.print_exc()

    elif args.single_sound_file:
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

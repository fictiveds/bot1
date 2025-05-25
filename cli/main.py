import argparse
import random
import time
import numpy as np
from pydub import AudioSegment # Хотя бы для type hinting, если понадобится

# Импорты из созданных модулей
from composer.composition import create_experimental_composition, get_random_params # noqa
from audio_engine.pydub_utils import ( # noqa
    get_sine_segment, get_square_segment, get_sawtooth_segment, get_noise_segment, # noqa
    apply_delay_to_segment, apply_filter_to_segment, apply_reverb_to_segment, # noqa
    apply_simplified_granular_effect # noqa
)
from utils.constants import SAMPLE_RATE
from nodal_engine.graph import SoundGraph
from nodal_engine.modules import SineOscillator, LFO, ADSREnvelope, SignalScalerOffset


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
        print(f"Генерация сложного примера нодального звука в файл: {args.nodal_example}")
        try:
            # 1. Создаем граф
            graph = SoundGraph()
            sample_rate = SAMPLE_RATE # Используем глобальную константу

            # 2. Создаем модули
            # LFO для модуляции частоты
            # Генерирует сигнал от -1 до 1 с частотой 0.5 Гц
            lfo_freq_mod = LFO(name="lfo_freq_mod", frequency=0.5, amplitude=1.0) 
            
            # Scaler/Offset для преобразования выхода LFO в диапазон частот
            # Базовая частота осциллятора будет 330 Гц.
            # LFO (-1 до 1) * 110 => -110 до 110 Гц (диапазон модуляции)
            # (LFO * 110) + 330 => от 220 Гц до 440 Гц
            freq_scaler = SignalScalerOffset(name="freq_scaler", scale=110.0, offset=330.0)

            # Осциллятор
            sine_osc = SineOscillator(name="sine_osc", frequency=330.0, amplitude=0.0) # Амплитуда будет управляться ADSR

            # ADSR огибающая для общей громкости
            # Attack: 0.1s, Decay: 0.2s, Sustain: 0.6, Release: 0.5s
            adsr_amp_env = ADSREnvelope(name="adsr_amp_env", 
                                        attack_time_sec=0.1, 
                                        decay_time_sec=0.2, 
                                        sustain_level=0.6, 
                                        release_time_sec=0.5)

            # 3. Добавляем модули в граф
            graph.add_module(lfo_freq_mod)
            graph.add_module(freq_scaler)
            graph.add_module(sine_osc)
            graph.add_module(adsr_amp_env)

            # 4. Соединяем модули
            # LFO -> Scaler (для частоты)
            graph.connect(source_module_name="lfo_freq_mod", source_output_name="value",
                          target_module_name="freq_scaler", target_input_name="input_signal")
            
            # Scaler -> Частота Осциллятора
            graph.connect(source_module_name="freq_scaler", source_output_name="value",
                          target_module_name="sine_osc", target_input_name="frequency")

            # ADSR -> Амплитуда Осциллятора
            graph.connect(source_module_name="adsr_amp_env", source_output_name="value",
                          target_module_name="sine_osc", target_input_name="amplitude")

            # 5. Устанавливаем мастер-выход
            graph.set_master_output("sine_osc")

            # 6. Триггеры для ADSR
            # Мы хотим, чтобы звук проиграл один раз на заданной длительности.
            # Для этого нужно будет как-то управлять trigger_on/trigger_off ADSR из графа
            # или передавать информацию о времени в ADSR.
            # Самый простой способ для этого примера: вызвать trigger_on() перед рендерингом,
            # и trigger_off() после определенного времени, если длительность рендера больше.
            # Однако, render_audio() сам по себе не имеет такой логики.
            #
            # ВАРИАНТ ДЛЯ ПРОСТОТЫ ПРИМЕРА:
            # Мы отрендерим звук чуть дольше, чем нужно для атаки-спада-сустейна,
            # и вызовем trigger_off до начала рендера той части, где должен быть release.
            # Это не идеально, но для демонстрации подойдет.
            # Более правильно было бы иметь модуль "Gate Sequencer" или передавать массив "гейт" сигналов в ADSR.
            
            duration = 3.0  # Общая длительность рендера в секундах
            
            # Запускаем ADSR в самом начале
            adsr_amp_env.trigger_on()
            
            # Мы не можем вызвать trigger_off() динамически во время рендера без изменений в SoundGraph.
            # Поэтому ADSR останется в Sustain фазе до конца, если gate_is_on=True.
            # Если мы хотим услышать Release, нам нужно установить gate_is_on=False в какой-то момент.
            # Для этого примера ADSR будет просто идти до Sustain и оставаться там,
            # так как trigger_off() не вызывается динамически во время рендеринга графа.
            # Чтобы услышать release, нужно было бы рендерить дольше и вызвать trigger_off()
            # *перед* вызовом render_audio() для той части, где ожидается release, что неудобно.
            #
            # Давайте сделаем так: ADSR будет активен (sustain) почти всю длительность,
            # а потом быстро затухнет. Мы можем имитировать это, установив очень короткое время release
            # и вызвав trigger_off() для ADSR *после* основного рендера, если бы мы рендерили по частям.
            #
            # Поскольку render_audio цельный, ADSR просто пройдет A-D-S. Если gate_is_on останется True,
            # он не войдет в Release. Если мы вызовем trigger_off() перед render_audio, то он сразу пойдет в Release.
            #
            # Для этого примера, чтобы ADSR отработала свой цикл (хотя бы ADS),
            # мы оставим adsr_amp_env.trigger_on(). Звук просто оборвется в конце sustain.
            # Чтобы был релиз, нужна более сложная логика управления gate.

            print(f"Параметры ADSR: A={adsr_amp_env.attack_time_sec}s, D={adsr_amp_env.decay_time_sec}s, S={adsr_amp_env.sustain_level}, R={adsr_amp_env.release_time_sec}s")

            # 7. Рендерим аудио
            output_audio_segment = graph.render_audio(duration, sample_rate)

            # 8. Сохраняем результат
            output_audio_segment.export(args.nodal_example, format="wav")
            print(f"Нодальный пример (ADSR + LFO->Freq) сохранен в: {args.nodal_example}")

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

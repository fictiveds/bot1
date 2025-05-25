# nodal_engine/modules.py
import numpy as np
import math # для pi
from .core import AudioModule, ControlModule 
from audio_engine.waveforms import generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise_wave
from audio_engine.effects import apply_delay, apply_filter, apply_reverb 
from audio_engine.pydub_utils import _numpy_to_segment, _segment_to_numpy, apply_simplified_granular_effect as pydub_granular_effect
from pydub import AudioSegment 
# from utils.constants import SAMPLE_RATE

class SineOscillator(AudioModule):
    """Осциллятор, генерирующий синусоидальную волну."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """
        Инициализирует осциллятор синусоидальной волны.

        Args:
            name (str): Имя модуля.
            frequency (float, optional): Начальная частота в Гц. Defaults to 440.0.
            amplitude (float, optional): Начальная амплитуда (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(name)
        self._frequency = frequency
        self._amplitude = amplitude
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def frequency(self) -> float:
        """Текущая частота осциллятора в Гц."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        self._frequency = float(value)

    @property
    def amplitude(self) -> float:
        """Текущая амплитуда осциллятора (0.0 до 1.0)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок синусоидальной волны. Частота и амплитуда могут управляться через входы 'frequency' и 'amplitude'."""
        freq_values = self.get_input_value('frequency', num_samples, sample_rate, default_value=self._frequency)
        amp_values = self.get_input_value('amplitude', num_samples, sample_rate, default_value=self._amplitude)

        current_freq = freq_values[0] if isinstance(freq_values, np.ndarray) and freq_values.size > 0 else self._frequency
        current_amp = amp_values[0] if isinstance(amp_values, np.ndarray) and amp_values.size > 0 else self._amplitude
        
        if isinstance(current_freq, np.ndarray): current_freq = current_freq.item()
        if isinstance(current_amp, np.ndarray): current_amp = current_amp.item()

        duration_sec = num_samples / sample_rate
        
        wave = generate_sine_wave(
            frequency=current_freq,
            duration=duration_sec,
            amplitude=current_amp,
            sample_rate=sample_rate
        )
        
        if len(wave) > num_samples:
            self.outputs['audio'] = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples:
            self.outputs['audio'] = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = wave.astype(np.float32)

class SquareOscillator(AudioModule):
    """Осциллятор, генерирующий прямоугольную волну."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """
        Инициализирует осциллятор прямоугольной волны.

        Args:
            name (str): Имя модуля.
            frequency (float, optional): Начальная частота в Гц. Defaults to 440.0.
            amplitude (float, optional): Начальная амплитуда (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(name)
        self._frequency = frequency
        self._amplitude = amplitude
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def frequency(self) -> float:
        """Текущая частота осциллятора в Гц."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        self._frequency = float(value)

    @property
    def amplitude(self) -> float:
        """Текущая амплитуда осциллятора (0.0 до 1.0)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок прямоугольной волны. Частота и амплитуда могут управляться через входы 'frequency' и 'amplitude'."""
        freq_values = self.get_input_value('frequency', num_samples, sample_rate, default_value=self._frequency)
        amp_values = self.get_input_value('amplitude', num_samples, sample_rate, default_value=self._amplitude)

        current_freq = freq_values[0] if isinstance(freq_values, np.ndarray) and freq_values.size > 0 else self._frequency
        current_amp = amp_values[0] if isinstance(amp_values, np.ndarray) and amp_values.size > 0 else self._amplitude

        if isinstance(current_freq, np.ndarray): current_freq = current_freq.item()
        if isinstance(current_amp, np.ndarray): current_amp = current_amp.item()
            
        duration_sec = num_samples / sample_rate
        
        wave = generate_square_wave(
            frequency=current_freq,
            duration=duration_sec,
            amplitude=current_amp,
            sample_rate=sample_rate
        )
        
        if len(wave) > num_samples:
            self.outputs['audio'] = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples:
            self.outputs['audio'] = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = wave.astype(np.float32)

class SawtoothOscillator(AudioModule):
    """Осциллятор, генерирующий пилообразную волну."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """
        Инициализирует осциллятор пилообразной волны.

        Args:
            name (str): Имя модуля.
            frequency (float, optional): Начальная частота в Гц. Defaults to 440.0.
            amplitude (float, optional): Начальная амплитуда (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(name)
        self._frequency = frequency
        self._amplitude = amplitude
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def frequency(self) -> float:
        """Текущая частота осциллятора в Гц."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        self._frequency = float(value)

    @property
    def amplitude(self) -> float:
        """Текущая амплитуда осциллятора (0.0 до 1.0)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок пилообразной волны. Частота и амплитуда могут управляться через входы 'frequency' и 'amplitude'."""
        freq_values = self.get_input_value('frequency', num_samples, sample_rate, default_value=self._frequency)
        amp_values = self.get_input_value('amplitude', num_samples, sample_rate, default_value=self._amplitude)

        current_freq = freq_values[0] if isinstance(freq_values, np.ndarray) and freq_values.size > 0 else self._frequency
        current_amp = amp_values[0] if isinstance(amp_values, np.ndarray) and amp_values.size > 0 else self._amplitude

        if isinstance(current_freq, np.ndarray): current_freq = current_freq.item()
        if isinstance(current_amp, np.ndarray): current_amp = current_amp.item()
            
        duration_sec = num_samples / sample_rate
        
        wave = generate_sawtooth_wave(
            frequency=current_freq,
            duration=duration_sec,
            amplitude=current_amp,
            sample_rate=sample_rate
        )
        
        if len(wave) > num_samples:
            self.outputs['audio'] = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples:
            self.outputs['audio'] = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = wave.astype(np.float32)

class NoiseGenerator(AudioModule):
    """Генератор, создающий белый шум."""
    def __init__(self, name: str, amplitude: float = 0.5):
        """
        Инициализирует генератор белого шума.

        Args:
            name (str): Имя модуля.
            amplitude (float, optional): Начальная амплитуда (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(name)
        self._amplitude = amplitude
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def amplitude(self) -> float:
        """Текущая амплитуда генератора шума (0.0 до 1.0)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок белого шума. Амплитуда может управляться через вход 'amplitude'."""
        amp_values = self.get_input_value('amplitude', num_samples, sample_rate, default_value=self._amplitude)
        
        current_amp = amp_values[0] if isinstance(amp_values, np.ndarray) and amp_values.size > 0 else self._amplitude

        if isinstance(current_amp, np.ndarray): current_amp = current_amp.item()

        duration_sec = num_samples / sample_rate
        
        wave = generate_noise_wave(
            duration=duration_sec,
            amplitude=current_amp,
            sample_rate=sample_rate
        )
        
        if len(wave) > num_samples:
            self.outputs['audio'] = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples:
            self.outputs['audio'] = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = wave.astype(np.float32)

# --- Классы Эффектов ---

class DelayEffect(AudioModule):
    """Применяет эффект задержки (delay) к аудиосигналу."""
    def __init__(self, name: str, delay_seconds: float = 0.5, decay_factor: float = 0.4):
        """
        Инициализирует модуль эффекта задержки.

        Args:
            name (str): Имя модуля.
            delay_seconds (float, optional): Начальное время задержки в секундах. Defaults to 0.5.
            decay_factor (float, optional): Начальный коэффициент затухания эха. Defaults to 0.4.
        """
        super().__init__(name)
        self._delay_seconds = delay_seconds
        self._decay_factor = decay_factor
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def delay_seconds(self) -> float:
        """Время задержки в секундах."""
        return self._delay_seconds

    @delay_seconds.setter
    def delay_seconds(self, value: float):
        self._delay_seconds = float(value)

    @property
    def decay_factor(self) -> float:
        """Коэффициент затухания эха (0.0 до 1.0)."""
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value: float):
        self._decay_factor = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект задержки. 
        Параметры 'delay_seconds' и 'decay_factor' могут управляться через входы."""
        audio_in_block = self.get_input_value('audio_in', num_samples, sample_rate, default_value=np.zeros(num_samples, dtype=np.float32))
        
        delay_sec_values = self.get_input_value('delay_seconds', num_samples, sample_rate, default_value=self._delay_seconds)
        decay_factor_values = self.get_input_value('decay_factor', num_samples, sample_rate, default_value=self._decay_factor)

        current_delay_sec = delay_sec_values[0] if isinstance(delay_sec_values, np.ndarray) and delay_sec_values.size > 0 else self._delay_seconds
        current_decay_factor = decay_factor_values[0] if isinstance(decay_factor_values, np.ndarray) and decay_factor_values.size > 0 else self._decay_factor
        
        if isinstance(current_delay_sec, np.ndarray): current_delay_sec = current_delay_sec.item()
        if isinstance(current_decay_factor, np.ndarray): current_decay_factor = current_decay_factor.item()
        
        processed_audio = apply_delay(
            wave_data=audio_in_block,
            sample_rate=sample_rate,
            delay_seconds=current_delay_sec,
            decay_factor=current_decay_factor
        )
        
        if len(processed_audio) > num_samples:
            self.outputs['audio'] = processed_audio[:num_samples].astype(np.float32)
        elif len(processed_audio) < num_samples:
            self.outputs['audio'] = np.pad(processed_audio, (0, num_samples - len(processed_audio)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = processed_audio.astype(np.float32)

class FilterEffect(AudioModule):
    """Применяет эффект фильтрации (ФНЧ или ФВЧ) к аудиосигналу."""
    def __init__(self, name: str, cutoff_hz: float = 1000.0, filter_type: str = 'lowpass', order: int = 5):
        """
        Инициализирует модуль эффекта фильтрации.

        Args:
            name (str): Имя модуля.
            cutoff_hz (float, optional): Начальная частота среза в Гц. Defaults to 1000.0.
            filter_type (str, optional): Тип фильтра ('lowpass' или 'highpass'). Defaults to 'lowpass'.
            order (int, optional): Порядок фильтра. Defaults to 5.
        """
        super().__init__(name)
        self._cutoff_hz = cutoff_hz
        if filter_type not in ['lowpass', 'highpass']:
            raise ValueError("filter_type должен быть 'lowpass' или 'highpass'")
        self._filter_type = filter_type
        self._order = order
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def cutoff_hz(self) -> float:
        """Частота среза фильтра в Гц."""
        return self._cutoff_hz

    @cutoff_hz.setter
    def cutoff_hz(self, value: float):
        self._cutoff_hz = float(value)

    @property
    def filter_type(self) -> str:
        """Тип фильтра ('lowpass' или 'highpass')."""
        return self._filter_type

    @filter_type.setter
    def filter_type(self, value: str):
        if value not in ['lowpass', 'highpass']:
            raise ValueError("filter_type должен быть 'lowpass' или 'highpass'")
        self._filter_type = value
        
    @property
    def order(self) -> int:
        """Порядок фильтра."""
        return self._order

    @order.setter
    def order(self, value: int):
        self._order = int(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект фильтрации. 
        Параметр 'cutoff_hz' может управляться через вход."""
        audio_in_block = self.get_input_value('audio_in', num_samples, sample_rate, default_value=np.zeros(num_samples, dtype=np.float32))
        cutoff_values = self.get_input_value('cutoff_hz', num_samples, sample_rate, default_value=self._cutoff_hz)
        
        current_cutoff = cutoff_values[0] if isinstance(cutoff_values, np.ndarray) and cutoff_values.size > 0 else self._cutoff_hz
        if isinstance(current_cutoff, np.ndarray): current_cutoff = current_cutoff.item()

        # filter_type и order пока не управляются через входы, используются внутренние значения
        processed_audio = apply_filter(
            wave_data=audio_in_block,
            sample_rate=sample_rate,
            cutoff_hz=current_cutoff,
            filter_type=self._filter_type,
            order=self._order
        )
        self.outputs['audio'] = processed_audio.astype(np.float32) # apply_filter возвращает массив нужной длины

class ReverbEffect(AudioModule):
    """Применяет эффект реверберации к аудиосигналу."""
    def __init__(self, name: str, number_of_delays: int = 5, max_delay_seconds: float = 0.5, overall_decay_factor: float = 0.6):
        """
        Инициализирует модуль эффекта реверберации.

        Args:
            name (str): Имя модуля.
            number_of_delays (int, optional): Начальное количество линий задержки. Defaults to 5.
            max_delay_seconds (float, optional): Начальное максимальное время задержки. Defaults to 0.5.
            overall_decay_factor (float, optional): Начальный общий коэффициент затухания. Defaults to 0.6.
        """
        super().__init__(name)
        self._number_of_delays = number_of_delays
        self._max_delay_seconds = max_delay_seconds
        self._overall_decay_factor = overall_decay_factor
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def number_of_delays(self) -> int:
        return self._number_of_delays
    @number_of_delays.setter
    def number_of_delays(self, value: int):
        self._number_of_delays = int(value)

    @property
    def max_delay_seconds(self) -> float:
        return self._max_delay_seconds
    @max_delay_seconds.setter
    def max_delay_seconds(self, value: float):
        self._max_delay_seconds = float(value)

    @property
    def overall_decay_factor(self) -> float:
        return self._overall_decay_factor
    @overall_decay_factor.setter
    def overall_decay_factor(self, value: float):
        self._overall_decay_factor = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект реверберации.
        Параметры могут управляться через входы 'number_of_delays', 'max_delay_seconds', 'overall_decay_factor'."""
        audio_in_block = self.get_input_value('audio_in', num_samples, sample_rate, default_value=np.zeros(num_samples, dtype=np.float32))
        
        # Для упрощения, пока параметры реверберации берутся из свойств, а не из управляющих входов
        # TODO: Добавить чтение параметров из self.get_input_value по аналогии с DelayEffect
        
        processed_audio = apply_reverb(
            wave_data=audio_in_block,
            sample_rate=sample_rate,
            number_of_delays=self._number_of_delays,
            max_delay_seconds=self._max_delay_seconds,
            overall_decay_factor=self._overall_decay_factor
        )
        
        if len(processed_audio) > num_samples:
            self.outputs['audio'] = processed_audio[:num_samples].astype(np.float32)
        elif len(processed_audio) < num_samples:
            self.outputs['audio'] = np.pad(processed_audio, (0, num_samples - len(processed_audio)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = processed_audio.astype(np.float32)

class GranularEffect(AudioModule):
    """Применяет упрощенный гранулярный эффект к аудиосигналу.
    (Временная реализация через конвертацию в/из AudioSegment)."""
    def __init__(self, name: str, grain_duration_ms: int = 50, density: float = 1.0, output_duration_factor: float = 1.0):
        """
        Инициализирует модуль гранулярного эффекта.

        Args:
            name (str): Имя модуля.
            grain_duration_ms (int, optional): Начальная длительность гранулы в мс. Defaults to 50.
            density (float, optional): Начальная плотность гранул. Defaults to 1.0.
            output_duration_factor (float, optional): Начальный множитель длительности выхода. Defaults to 1.0.
        """
        super().__init__(name)
        self._grain_duration_ms = grain_duration_ms
        self._density = density
        self._output_duration_factor = output_duration_factor
        self.outputs['audio'] = np.zeros(0, dtype=np.float32)

    @property
    def grain_duration_ms(self) -> int:
        return self._grain_duration_ms
    @grain_duration_ms.setter
    def grain_duration_ms(self, value: int):
        self._grain_duration_ms = int(value)

    @property
    def density(self) -> float:
        return self._density
    @density.setter
    def density(self, value: float):
        self._density = float(value)

    @property
    def output_duration_factor(self) -> float:
        return self._output_duration_factor
    @output_duration_factor.setter
    def output_duration_factor(self, value: float):
        self._output_duration_factor = float(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя гранулярный эффект.
        Параметры могут управляться через входы 'grain_duration_ms', 'density', 'output_duration_factor'."""
        audio_in_block = self.get_input_value('audio_in', num_samples, sample_rate, default_value=np.zeros(num_samples, dtype=np.float32))
        
        # TODO: Добавить чтение параметров из self.get_input_value
        # Пока используем внутренние значения

        if audio_in_block.size == 0: # Если входной блок пустой
            self.outputs['audio'] = np.zeros(num_samples, dtype=np.float32)
            return

        audio_segment_in = _numpy_to_segment(audio_in_block, sample_rate)
        
        processed_segment_out = pydub_granular_effect(
            audio_segment_in,
            grain_duration_ms=self._grain_duration_ms,
            density=self._density,
            output_duration_factor=self._output_duration_factor
        )
        
        processed_audio_np = _segment_to_numpy(processed_segment_out)
        
        # Регулировка длины результата
        if len(processed_audio_np) > num_samples:
            self.outputs['audio'] = processed_audio_np[:num_samples].astype(np.float32)
        elif len(processed_audio_np) < num_samples:
            # Если результат короче, дополняем нулями до нужной длины
            # Это может быть неидеально для гранулярного синтеза, т.к. он может менять длительность
            self.outputs['audio'] = np.pad(processed_audio_np, (0, num_samples - len(processed_audio_np)), 'constant').astype(np.float32)
        else:
            self.outputs['audio'] = processed_audio_np.astype(np.float32)

# --- Класс LFO ---
class LFO(ControlModule):
    """Низкочастотный осциллятор (LFO) для генерации управляющих сигналов."""
    def __init__(self, name: str, frequency: float = 1.0, amplitude: float = 1.0, initial_phase_degrees: float = 0.0):
        """
        Инициализирует низкочастотный осциллятор (LFO).

        Args:
            name (str): Имя модуля.
            frequency (float, optional): Начальная частота LFO в Гц. Defaults to 1.0.
            amplitude (float, optional): Начальная амплитуда выходного сигнала LFO. Defaults to 1.0.
            initial_phase_degrees (float, optional): Начальная фаза LFO в градусах. Defaults to 0.0.
        """
        super().__init__(name)
        self._frequency = frequency  # Частота LFO в Гц
        self._amplitude = amplitude  # Амплитуда выходного сигнала LFO
        self._current_phase_rad = math.radians(initial_phase_degrees)  # Внутренняя текущая фаза в радианах
        
        # Выход 'value' уже инициализирован в ControlModule как 0.0
        # Мы будем обновлять его массивом в process_block
        self.outputs['value'] = np.zeros(0, dtype=np.float32)


    @property
    def frequency(self) -> float:
        """Частота LFO в Герцах."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        self._frequency = float(value)

    @property
    def amplitude(self) -> float:
        """Амплитуда выходного сигнала LFO (обычно от 0.0 до 1.0, но может быть и больше/меньше)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = float(value)

    @property
    def initial_phase_degrees(self) -> float:
        """Начальная фаза LFO в градусах. Установка этого значения сбрасывает текущую фазу."""
        # Возвращаем текущую фазу, преобразованную в градусы, как представление начальной/текущей точки.
        # Это свойство, по сути, позволяет "перезапустить" LFO с новой фазы.
        return math.degrees(self._current_phase_rad)

    @initial_phase_degrees.setter
    def initial_phase_degrees(self, value: float):
        # При изменении начальной фазы, сбрасываем текущую фазу на это значение
        self._current_phase_rad = math.radians(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """
        Генерирует блок управляющего сигнала LFO (синусоида).
        Обновляет внутреннюю фазу для обеспечения непрерывности сигнала между блоками.
        """
        # Для LFO параметры frequency и amplitude пока не модулируются другими сигналами,
        # а берутся из его собственных свойств.
        # В будущем можно добавить входы 'frequency_in', 'amplitude_in'.
        current_lfo_freq = self._frequency
        current_lfo_amplitude = self._amplitude

        output_block = np.zeros(num_samples, dtype=np.float32)
        phase_step = (2 * math.pi * current_lfo_freq) / sample_rate
        
        current_phase = self._current_phase_rad
        for i in range(num_samples):
            output_block[i] = current_lfo_amplitude * math.sin(current_phase)
            current_phase += phase_step
        
        self._current_phase_rad = current_phase % (2 * math.pi) # Сохраняем и нормализуем фазу для следующего блока

        self.outputs['value'] = output_block

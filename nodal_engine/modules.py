# nodal_engine/modules.py
import numpy as np
import math # для pi
from enum import Enum, auto
from .core import AudioModule, ControlModule, InputConnector, OutputConnector, ModulationType
from audio_engine.waveforms import generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise_wave
from audio_engine.effects import apply_delay, apply_filter, apply_reverb 
from audio_engine.pydub_utils import _numpy_to_segment, _segment_to_numpy, apply_simplified_granular_effect as pydub_granular_effect
# from pydub import AudioSegment # Больше не нужен здесь напрямую, если pydub_utils справляется

class EnvelopeState(Enum):
    """Состояния ADSR-огибающей."""
    IDLE = auto()
    ATTACK = auto()
    DECAY = auto()
    SUSTAIN = auto()
    RELEASE = auto()

class SineOscillator(AudioModule):
    """Осциллятор, генерирующий синусоидальную волну (v2 с коннекторами)."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """
        Инициализирует осциллятор синусоидальной волны.

        Args:
            name (str): Имя модуля.
            frequency (float, optional): Начальная частота в Гц. Defaults to 440.0.
            amplitude (float, optional): Начальная амплитуда (0.0-1.0). Defaults to 0.5.
        """
        super().__init__(name)
        self.frequency_in = InputConnector(name='frequency', module_owner=self, default_value=frequency, modulation_type=ModulationType.REPLACE)
        self.amplitude_in = InputConnector(name='amplitude', module_owner=self, default_value=amplitude, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32) # Инициализация выходного буфера

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок синусоидальной волны, используя значения с входных коннекторов."""
        freq_values = self.frequency_in.get_value(num_samples, sample_rate)
        amp_values = self.amplitude_in.get_value(num_samples, sample_rate)
        
        # Для осцилляторов обычно используется одно значение частоты/амплитуды на блок,
        # или первое значение из модулирующего массива.
        # Более сложная логика (например, обработка каждого сэмпла с разной частотой) здесь не реализована.
        current_freq = freq_values[0] 
        current_amp = amp_values[0]  

        duration_sec = num_samples / sample_rate
        
        wave = generate_sine_wave(
            frequency=current_freq,
            duration=duration_sec,
            amplitude=current_amp,
            sample_rate=sample_rate
        )
        
        # Обеспечение корректной длины и типа
        if len(wave) > num_samples:
            processed_wave = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples:
            processed_wave = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else:
            processed_wave = wave.astype(np.float32)
        
        self.audio_out.value = processed_wave

class SquareOscillator(AudioModule):
    """Осциллятор, генерирующий прямоугольную волну (v2 с коннекторами)."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """Инициализирует осциллятор прямоугольной волны."""
        super().__init__(name)
        self.frequency_in = InputConnector(name='frequency', module_owner=self, default_value=frequency, modulation_type=ModulationType.REPLACE)
        self.amplitude_in = InputConnector(name='amplitude', module_owner=self, default_value=amplitude, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок прямоугольной волны."""
        freq_values = self.frequency_in.get_value(num_samples, sample_rate)
        amp_values = self.amplitude_in.get_value(num_samples, sample_rate)
        current_freq = freq_values[0]
        current_amp = amp_values[0]
        duration_sec = num_samples / sample_rate
        wave = generate_square_wave(frequency=current_freq, duration=duration_sec, amplitude=current_amp, sample_rate=sample_rate)
        if len(wave) > num_samples: self.audio_out.value = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples: self.audio_out.value = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else: self.audio_out.value = wave.astype(np.float32)

class SawtoothOscillator(AudioModule):
    """Осциллятор, генерирующий пилообразную волну (v2 с коннекторами)."""
    def __init__(self, name: str, frequency: float = 440.0, amplitude: float = 0.5):
        """Инициализирует осциллятор пилообразной волны."""
        super().__init__(name)
        self.frequency_in = InputConnector(name='frequency', module_owner=self, default_value=frequency, modulation_type=ModulationType.REPLACE)
        self.amplitude_in = InputConnector(name='amplitude', module_owner=self, default_value=amplitude, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок пилообразной волны."""
        freq_values = self.frequency_in.get_value(num_samples, sample_rate)
        amp_values = self.amplitude_in.get_value(num_samples, sample_rate)
        current_freq = freq_values[0]
        current_amp = amp_values[0]
        duration_sec = num_samples / sample_rate
        wave = generate_sawtooth_wave(frequency=current_freq, duration=duration_sec, amplitude=current_amp, sample_rate=sample_rate)
        if len(wave) > num_samples: self.audio_out.value = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples: self.audio_out.value = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else: self.audio_out.value = wave.astype(np.float32)

class NoiseGenerator(AudioModule):
    """Генератор, создающий белый шум (v2 с коннекторами)."""
    def __init__(self, name: str, amplitude: float = 0.5):
        """Инициализирует генератор белого шума."""
        super().__init__(name)
        self.amplitude_in = InputConnector(name='amplitude', module_owner=self, default_value=amplitude, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок белого шума."""
        amp_values = self.amplitude_in.get_value(num_samples, sample_rate)
        current_amp = amp_values[0]
        duration_sec = num_samples / sample_rate
        wave = generate_noise_wave(duration=duration_sec, amplitude=current_amp, sample_rate=sample_rate)
        if len(wave) > num_samples: self.audio_out.value = wave[:num_samples].astype(np.float32)
        elif len(wave) < num_samples: self.audio_out.value = np.pad(wave, (0, num_samples - len(wave)), 'constant').astype(np.float32)
        else: self.audio_out.value = wave.astype(np.float32)

# --- Классы Эффектов (v2 с коннекторами) ---

class DelayEffect(AudioModule):
    """Применяет эффект задержки (delay) к аудиосигналу (v2 с коннекторами)."""
    def __init__(self, name: str, delay_seconds: float = 0.5, decay_factor: float = 0.4):
        """Инициализирует модуль эффекта задержки."""
        super().__init__(name)
        self.audio_in = InputConnector(name='audio_in', module_owner=self, default_value=np.zeros(1, dtype=np.float32)) # default_value будет растянут до num_samples в get_value
        self.delay_seconds_in = InputConnector(name='delay_seconds', module_owner=self, default_value=delay_seconds, modulation_type=ModulationType.REPLACE)
        self.decay_factor_in = InputConnector(name='decay_factor', module_owner=self, default_value=decay_factor, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект задержки."""
        audio_in_block = self.audio_in.get_value(num_samples, sample_rate)
        delay_sec_values = self.delay_seconds_in.get_value(num_samples, sample_rate)
        decay_factor_values = self.decay_factor_in.get_value(num_samples, sample_rate)
        
        current_delay_sec = delay_sec_values[0]
        current_decay_factor = decay_factor_values[0]
        
        processed_audio = apply_delay(wave_data=audio_in_block, sample_rate=sample_rate, delay_seconds=current_delay_sec, decay_factor=current_decay_factor)
        
        if len(processed_audio) > num_samples: self.audio_out.value = processed_audio[:num_samples].astype(np.float32)
        elif len(processed_audio) < num_samples: self.audio_out.value = np.pad(processed_audio, (0, num_samples - len(processed_audio)), 'constant').astype(np.float32)
        else: self.audio_out.value = processed_audio.astype(np.float32)

class FilterEffect(AudioModule):
    """Применяет эффект фильтрации (ФНЧ или ФВЧ) к аудиосигналу (v2 с коннекторами)."""
    def __init__(self, name: str, cutoff_hz: float = 1000.0, filter_type: str = 'lowpass', order: int = 5):
        """Инициализирует модуль эффекта фильтрации."""
        super().__init__(name)
        self.audio_in = InputConnector(name='audio_in', module_owner=self, default_value=np.zeros(1, dtype=np.float32))
        self.cutoff_hz_in = InputConnector(name='cutoff_hz', module_owner=self, default_value=cutoff_hz, modulation_type=ModulationType.REPLACE)
        
        if filter_type not in ['lowpass', 'highpass']: raise ValueError("filter_type должен быть 'lowpass' или 'highpass'")
        self._filter_type = filter_type # Оставляем как property, не модулируемый через InputConnector
        self._order = order             # Оставляем как property, не модулируемый через InputConnector
        
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    @property
    def filter_type(self) -> str: """Тип фильтра ('lowpass' или 'highpass')."""
        return self._filter_type
    @filter_type.setter
    def filter_type(self, value: str):
        if value not in ['lowpass', 'highpass']: raise ValueError("filter_type должен быть 'lowpass' или 'highpass'")
        self._filter_type = value
        
    @property
    def order(self) -> int: """Порядок фильтра."""
        return self._order
    @order.setter
    def order(self, value: int): self._order = int(value)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект фильтрации."""
        audio_in_block = self.audio_in.get_value(num_samples, sample_rate)
        cutoff_values = self.cutoff_hz_in.get_value(num_samples, sample_rate)
        current_cutoff = cutoff_values[0]
        
        processed_audio = apply_filter(wave_data=audio_in_block, sample_rate=sample_rate, cutoff_hz=current_cutoff, filter_type=self._filter_type, order=self._order)
        self.audio_out.value = processed_audio.astype(np.float32)

class ReverbEffect(AudioModule):
    """Применяет эффект реверберации к аудиосигналу (v2 с коннекторами)."""
    def __init__(self, name: str, number_of_delays: int = 5, max_delay_seconds: float = 0.5, overall_decay_factor: float = 0.6):
        """Инициализирует модуль эффекта реверберации."""
        super().__init__(name)
        self.audio_in = InputConnector(name='audio_in', module_owner=self, default_value=np.zeros(1, dtype=np.float32))
        self.number_of_delays_in = InputConnector(name='number_of_delays', module_owner=self, default_value=float(number_of_delays), modulation_type=ModulationType.REPLACE)
        self.max_delay_seconds_in = InputConnector(name='max_delay_seconds', module_owner=self, default_value=max_delay_seconds, modulation_type=ModulationType.REPLACE)
        self.overall_decay_factor_in = InputConnector(name='overall_decay_factor', module_owner=self, default_value=overall_decay_factor, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя эффект реверберации."""
        audio_in_block = self.audio_in.get_value(num_samples, sample_rate)
        num_delays_vals = self.number_of_delays_in.get_value(num_samples, sample_rate)
        max_delay_vals = self.max_delay_seconds_in.get_value(num_samples, sample_rate)
        decay_vals = self.overall_decay_factor_in.get_value(num_samples, sample_rate)

        current_num_delays = int(num_delays_vals[0])
        current_max_delay = max_delay_vals[0]
        current_decay = decay_vals[0]
        
        processed_audio = apply_reverb(wave_data=audio_in_block, sample_rate=sample_rate, number_of_delays=current_num_delays, max_delay_seconds=current_max_delay, overall_decay_factor=current_decay)
        
        if len(processed_audio) > num_samples: self.audio_out.value = processed_audio[:num_samples].astype(np.float32)
        elif len(processed_audio) < num_samples: self.audio_out.value = np.pad(processed_audio, (0, num_samples - len(processed_audio)), 'constant').astype(np.float32)
        else: self.audio_out.value = processed_audio.astype(np.float32)

class GranularEffect(AudioModule):
    """Применяет упрощенный гранулярный эффект (v2 с коннекторами)."""
    def __init__(self, name: str, grain_duration_ms: int = 50, density: float = 1.0, output_duration_factor: float = 1.0):
        """Инициализирует модуль гранулярного эффекта."""
        super().__init__(name)
        self.audio_in = InputConnector(name='audio_in', module_owner=self, default_value=np.zeros(1, dtype=np.float32))
        self.grain_duration_ms_in = InputConnector(name='grain_duration_ms', module_owner=self, default_value=float(grain_duration_ms), modulation_type=ModulationType.REPLACE)
        self.density_in = InputConnector(name='density', module_owner=self, default_value=density, modulation_type=ModulationType.REPLACE)
        self.output_duration_factor_in = InputConnector(name='output_duration_factor', module_owner=self, default_value=output_duration_factor, modulation_type=ModulationType.REPLACE)
        self.audio_out = OutputConnector(name='audio', module_owner=self)
        self.audio_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок аудио, применяя гранулярный эффект."""
        audio_in_block_np = self.audio_in.get_value(num_samples, sample_rate)
        grain_dur_vals = self.grain_duration_ms_in.get_value(num_samples, sample_rate)
        density_vals = self.density_in.get_value(num_samples, sample_rate)
        out_factor_vals = self.output_duration_factor_in.get_value(num_samples, sample_rate)

        current_grain_dur = int(grain_dur_vals[0])
        current_density = density_vals[0]
        current_out_factor = out_factor_vals[0]

        if audio_in_block_np.size == 0:
            self.audio_out.value = np.zeros(num_samples, dtype=np.float32)
            return

        audio_segment_in = _numpy_to_segment(audio_in_block_np, sample_rate)
        processed_segment_out = pydub_granular_effect(audio_segment_in, grain_duration_ms=current_grain_dur, density=current_density, output_duration_factor=current_out_factor)
        processed_audio_np = _segment_to_numpy(processed_segment_out)
        
        if len(processed_audio_np) > num_samples: self.audio_out.value = processed_audio_np[:num_samples].astype(np.float32)
        elif len(processed_audio_np) < num_samples: self.audio_out.value = np.pad(processed_audio_np, (0, num_samples - len(processed_audio_np)), 'constant').astype(np.float32)
        else: self.audio_out.value = processed_audio_np.astype(np.float32)

# --- Класс LFO (v2 с коннекторами) ---
class LFO(ControlModule):
    """Низкочастотный осциллятор (LFO) для генерации управляющих сигналов (v2)."""
    def __init__(self, name: str, frequency: float = 1.0, amplitude: float = 1.0, initial_phase_degrees: float = 0.0):
        """Инициализирует LFO."""
        super().__init__(name)
        self.frequency_in = InputConnector(name='frequency', module_owner=self, default_value=frequency, modulation_type=ModulationType.REPLACE)
        self.amplitude_in = InputConnector(name='amplitude', module_owner=self, default_value=amplitude, modulation_type=ModulationType.REPLACE)
        self.value_out = OutputConnector(name='value', module_owner=self)
        self.value_out.value = np.zeros(0, dtype=np.float32)
        
        self._current_phase_rad = math.radians(initial_phase_degrees)
        self._initial_phase_degrees = initial_phase_degrees # Сохраняем для property

    @property
    def initial_phase_degrees(self) -> float:
        """Начальная фаза LFO в градусах. Установка этого значения сбрасывает текущую фазу."""
        return self._initial_phase_degrees
    @initial_phase_degrees.setter
    def initial_phase_degrees(self, value: float):
        self._initial_phase_degrees = float(value)
        self._current_phase_rad = math.radians(self._initial_phase_degrees)


    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок управляющего сигнала LFO (синусоида)."""
        freq_values = self.frequency_in.get_value(num_samples, sample_rate)
        amp_values = self.amplitude_in.get_value(num_samples, sample_rate)
        
        current_lfo_freq = freq_values[0]
        current_lfo_amplitude = amp_values[0]

        output_block = np.zeros(num_samples, dtype=np.float32)
        phase_step = (2 * math.pi * current_lfo_freq) / sample_rate
        current_phase = self._current_phase_rad
        for i in range(num_samples):
            output_block[i] = current_lfo_amplitude * math.sin(current_phase)
            current_phase += phase_step
        self._current_phase_rad = current_phase % (2 * math.pi)
        self.value_out.value = output_block

# --- Класс ADSR Envelope (v2 с коннекторами) ---
class ADSREnvelope(ControlModule):
    """Генерирует ADSR-огибающую (v2). Параметры A,D,S,R устанавливаются через properties."""
    def __init__(self, name: str, attack_time_sec: float = 0.1, decay_time_sec: float = 0.1, sustain_level: float = 0.7, release_time_sec: float = 0.5):
        """Инициализирует ADSR-огибающую."""
        super().__init__(name)
        self.value_out = OutputConnector(name='value', module_owner=self)
        self.value_out.value = np.zeros(0, dtype=np.float32)

        self._attack_time_sec = max(0.001, attack_time_sec) 
        self._decay_time_sec = max(0.001, decay_time_sec)
        self._sustain_level = np.clip(sustain_level, 0.0, 1.0)
        self._release_time_sec = max(0.001, release_time_sec)

        self._state = EnvelopeState.IDLE
        self._current_level = 0.0
        self._gate_is_on = False 

    @property
    def attack_time(self) -> float: """Время атаки в секундах."""
        return self._attack_time_sec
    @attack_time.setter
    def attack_time(self, value: float): self._attack_time_sec = max(0.001, value)

    @property
    def decay_time(self) -> float: """Время спада до уровня поддержки в секундах."""
        return self._decay_time_sec
    @decay_time.setter
    def decay_time(self, value: float): self._decay_time_sec = max(0.001, value)

    @property
    def sustain(self) -> float: """Уровень поддержки (0.0 до 1.0)."""
        return self._sustain_level
    @sustain.setter
    def sustain(self, value: float): self._sustain_level = np.clip(value, 0.0, 1.0)
    
    @property
    def release_time(self) -> float: """Время затухания (после отпускания клавиши) в секундах."""
        return self._release_time_sec
    @release_time.setter
    def release_time(self, value: float): self._release_time_sec = max(0.001, value)

    def trigger_on(self):
        """Запускает огибающую (эквивалент нажатия клавиши)."""
        self._gate_is_on = True
        self._state = EnvelopeState.ATTACK

    def trigger_off(self):
        """Инициирует фазу затухания Release (эквивалент отпускания клавиши)."""
        self._gate_is_on = False
        if self._state != EnvelopeState.IDLE: self._state = EnvelopeState.RELEASE

    def process_block(self, num_samples: int, sample_rate: int):
        """Генерирует блок значений огибающей ADSR."""
        output_block = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            if self._state == EnvelopeState.IDLE: self._current_level = 0.0
            elif self._state == EnvelopeState.ATTACK:
                attack_samples = self._attack_time_sec * sample_rate
                increment = (1.0 - self._current_level) / (attack_samples * (1.0 - self._current_level + 1e-9) + 1e-9) if attack_samples > 0 else 1.0
                self._current_level += increment
                if self._current_level >= 1.0: self._current_level = 1.0; self._state = EnvelopeState.DECAY
            elif self._state == EnvelopeState.DECAY:
                decay_samples = self._decay_time_sec * sample_rate
                if self._current_level > self._sustain_level:
                    decrement = (self._current_level - self._sustain_level) / (decay_samples + 1e-9) if decay_samples > 0 else (self._current_level - self._sustain_level)
                    self._current_level -= decrement
                if self._current_level <= self._sustain_level: self._current_level = self._sustain_level; self._state = EnvelopeState.SUSTAIN
            elif self._state == EnvelopeState.SUSTAIN:
                self._current_level = self._sustain_level
                if not self._gate_is_on: self._state = EnvelopeState.RELEASE
            elif self._state == EnvelopeState.RELEASE:
                release_samples = self._release_time_sec * sample_rate
                decrement = self._current_level / (release_samples + 1e-9) if release_samples > 0 else self._current_level
                self._current_level -= decrement
                if self._current_level <= 0.0: self._current_level = 0.0; self._state = EnvelopeState.IDLE
            output_block[i] = self._current_level
        self.value_out.value = output_block

# --- Класс SignalScalerOffset (v2 с коннекторами) ---
class SignalScalerOffset(ControlModule):
    """Масштабирует и смещает входной управляющий сигнал (v2)."""
    def __init__(self, name: str, scale: float = 1.0, offset: float = 0.0):
        """Инициализирует модуль масштабирования и смещения сигнала."""
        super().__init__(name)
        self.input_signal_in = InputConnector(name='input_signal', module_owner=self, default_value=0.0, modulation_type=ModulationType.REPLACE)
        self.scale_in = InputConnector(name='scale', module_owner=self, default_value=scale, modulation_type=ModulationType.REPLACE)
        self.offset_in = InputConnector(name='offset', module_owner=self, default_value=offset, modulation_type=ModulationType.REPLACE)
        self.value_out = OutputConnector(name='value', module_owner=self)
        self.value_out.value = np.zeros(0, dtype=np.float32)

    def process_block(self, num_samples: int, sample_rate: int):
        """Обрабатывает блок, применяя масштабирование и смещение."""
        input_block = self.input_signal_in.get_value(num_samples, sample_rate)
        scale_values = self.scale_in.get_value(num_samples, sample_rate)
        offset_values = self.offset_in.get_value(num_samples, sample_rate)
        
        # Используем поэлементные операции, если scale/offset являются массивами (например, от LFO)
        # или скалярные, если они постоянны (get_value вернет растянутый массив)
        output_block = (input_block * scale_values) + offset_values
        self.value_out.value = output_block.astype(np.float32)

```

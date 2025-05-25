# nodal_engine/graph.py
import numpy as np
import math # Добавлен импорт math
from pydub import AudioSegment
from .core import Module, AudioModule, ControlModule
from audio_engine.pydub_utils import _numpy_to_segment 
from utils.constants import SAMPLE_RATE as DEFAULT_SAMPLE_RATE

class SoundGraph:
    """
    Управляет коллекцией звуковых модулей (узлов), их соединениями 
    и процессом рендеринга аудио из графа. Позволяет создавать сложные
    звуковые структуры путем соединения различных генераторов и эффектов.
    """
    def __init__(self):
        """
        Инициализирует пустой звуковой граф.
        - `modules`: Словарь для хранения модулей {имя_модуля: экземпляр_модуля}.
        - `_processing_order`: Внутренний список, определяющий порядок обработки модулей.
        - `master_output_module_name`: Имя модуля, чей аудиовыход считается основным выходом графа.
        """
        self.modules = {} 
        self._processing_order = [] 
        self.master_output_module_name = None 

    def add_module(self, module: Module):
        """
        Добавляет звуковой модуль в граф.

        Args:
            module (Module): Экземпляр модуля (наследник `Module`, `AudioModule` или `ControlModule`).

        Raises:
            ValueError: Если модуль с таким именем уже существует в графе.
        """
        if module.name in self.modules:
            raise ValueError(f"Модуль с именем '{module.name}' уже существует в графе.")
        self.modules[module.name] = module
        # Простая эвристика для порядка обработки: сначала Control, потом Audio
        # В будущем это можно улучшить, например, топологической сортировкой графа зависимостей.
        if isinstance(module, ControlModule):
            self._processing_order.insert(0, module.name) # Control модули обрабатываются первыми
        else:
            self._processing_order.append(module.name) # Audio модули обрабатываются позже
        print(f"Модуль '{module.name}' ({type(module).__name__}) добавлен в граф.")

    def connect(self, source_module_name: str, source_output_name: str, 
                target_module_name: str, target_input_name: str):
        """
        Соединяет указанный выход одного модуля с указанным входом другого модуля.

        Args:
            source_module_name (str): Имя модуля-источника.
            source_output_name (str): Имя выхода на модуле-источнике.
            target_module_name (str): Имя целевого модуля.
            target_input_name (str): Имя входа на целевом модуле.

        Raises:
            ValueError: Если один из модулей не найден в графе.
                        (Ошибка из `target_module.connect` если выход не найден у источника).
        """
        if source_module_name not in self.modules:
            raise ValueError(f"Модуль-источник '{source_module_name}' не найден в графе.")
        if target_module_name not in self.modules:
            raise ValueError(f"Целевой модуль '{target_module_name}' не найден в графе.")

        source_module = self.modules[source_module_name]
        target_module = self.modules[target_module_name]

        target_module.connect(target_input_name, source_module, source_output_name)

    def set_master_output(self, module_name: str):
        """
        Устанавливает модуль, чей выход 'audio' будет считаться мастер-выходом графа.
        Этот модуль должен быть экземпляром `AudioModule`.

        Args:
            module_name (str): Имя модуля, назначаемого мастер-выходом.

        Raises:
            ValueError: Если модуль с таким именем не найден или не является `AudioModule`.
        """
        if module_name not in self.modules or not isinstance(self.modules[module_name], AudioModule):
            raise ValueError(f"Модуль '{module_name}' не найден или не является AudioModule.")
        self.master_output_module_name = module_name
        print(f"Мастер-выход графа установлен на модуль '{module_name}'.")

    def render_audio(self, duration_seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioSegment:
        """
        Рендерит аудио из графа указанной длительности и частоты дискретизации.
        Обработка происходит блоками. Модули обрабатываются в определенном порядке.
        Аудио с выхода мастер-модуля (если указан) или сумма выходов всех AudioModule 
        (если мастер не указан) формирует итоговый результат.

        Args:
            duration_seconds (float): Желаемая длительность аудио в секундах.
            sample_rate (int, optional): Частота дискретизации. Defaults to DEFAULT_SAMPLE_RATE.

        Returns:
            AudioSegment: Сгенерированный и смешанный аудиосегмент.
                         Возвращает пустой `AudioSegment`, если `duration_seconds` <= 0.
        """
        total_samples = int(duration_seconds * sample_rate)
        if total_samples <= 0:
            return AudioSegment.empty()

        # Инициализация/очистка выходных буферов модулей перед новым рендерингом
        for module_name in self._processing_order:
            module = self.modules[module_name]
            for output_name in module.outputs: # module.outputs это словарь
                if output_name == 'audio' and isinstance(module, AudioModule):
                     module.outputs[output_name] = np.zeros(0, dtype=np.float32) 
                elif output_name == 'value' and isinstance(module, ControlModule):
                     module.outputs[output_name] = 0.0 
                else: 
                     module.outputs[output_name] = None


        final_output_accumulator = np.zeros(total_samples, dtype=np.float32)
        
        block_size = 256 # Размер блока обработки в сэмплах (можно сделать настраиваемым)

        num_blocks = math.ceil(total_samples / block_size)

        print(f"Начало рендеринга: {duration_seconds} сек, {sample_rate} Гц, {total_samples} сэмплов, {num_blocks} блоков по {block_size} сэмплов.")

        for i in range(num_blocks):
            current_pos = i * block_size
            samples_in_block = min(block_size, total_samples - current_pos)
            if samples_in_block <= 0:
                break
            
            # 1. Обработка всех модулей в установленном порядке
            for module_name in self._processing_order:
                module = self.modules[module_name]
                module.process_block(samples_in_block, sample_rate)

            # 2. Сбор выхода с мастер-модуля или суммирование всех аудио выходов
            block_audio_data = np.zeros(samples_in_block, dtype=np.float32)
            
            if self.master_output_module_name:
                master_module = self.modules[self.master_output_module_name]
                audio_out = master_module.outputs.get('audio')
                if isinstance(audio_out, np.ndarray) and audio_out.size == samples_in_block:
                    block_audio_data = audio_out.astype(np.float32) # Убедимся, что тип float32
                # else: # Опциональная отладка
                #     print(f"Предупреждение: Мастер-выход '{self.master_output_module_name}' (блок {i}) не произвел корректный аудио блок (размер {audio_out.size if isinstance(audio_out, np.ndarray) else 'None'}, ожидалось {samples_in_block}).")

            else: # Если мастер не задан, суммируем выходы всех AudioModule (примитивное авто-микширование)
                for module_name in self._processing_order:
                    module = self.modules[module_name]
                    if isinstance(module, AudioModule):
                        audio_out = module.outputs.get('audio')
                        if isinstance(audio_out, np.ndarray) and audio_out.size == samples_in_block:
                            block_audio_data += audio_out.astype(np.float32)
                        # else: # Опциональная отладка
                        #     print(f"Предупреждение: Модуль '{module.name}' (блок {i}) не произвел корректный аудио блок для авто-микширования (размер {audio_out.size if isinstance(audio_out, np.ndarray) else 'None'}, ожидалось {samples_in_block}).")
            
            final_output_accumulator[current_pos : current_pos + samples_in_block] = block_audio_data

        # Опциональная нормализация всего трека
        # peak_value = np.max(np.abs(final_output_accumulator))
        # if peak_value > 1.0: # Нормализуем, только если есть клиппинг
        #    final_output_accumulator /= peak_value
        # final_output_accumulator *= 0.98 # Небольшой запас по громкости (-0.17 dB)

        print("Рендеринг завершен.")
        return _numpy_to_segment(final_output_accumulator, sample_rate)


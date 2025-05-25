# nodal_engine/core.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Any, Dict # Добавлен Dict
from enum import Enum, auto

class ModulationType(Enum):
    """
    Определяет, как значение с подключенного входа модулятора
    взаимодействует с базовым значением параметра целевого модуля.
    """
    REPLACE = auto()  # Заменить базовое значение значением с входа (по умолчанию)
    ADD = auto()      # Прибавить значение с входа к базовому значению
    MULTIPLY = auto() # Умножить базовое значение на значение с входа

class OutputConnector:
    """
    Представляет выходной порт (коннектор) модуля.
    Хранит последнее вычисленное значение или блок значений (NumPy массив).
    """
    def __init__(self, name: str, module_owner: 'Module'):
        """
        Инициализирует выходной коннектор.

        Args:
            name (str): Имя выходного коннектора (например, 'audio', 'value', 'frequency_out').
            module_owner (Module): Модуль, которому принадлежит этот коннектор.
        """
        self.name = name
        self.module_owner = module_owner
        self.value: Optional[np.ndarray] = None # Хранит NumPy массив или None

    def __repr__(self):
        return f"<OutputConnector name='{self.name}' of module='{self.module_owner.name}'>"

class InputConnector:
    """
    Представляет входной порт (коннектор) модуля.
    Отвечает за получение значения от подключенного `OutputConnector`,
    применение масштабирования/смещения к этому значению, и его комбинацию
    с локальным значением по умолчанию параметра, к которому он относится.
    """
    def __init__(self, name: str, module_owner: 'Module', 
                 default_value: Union[float, np.ndarray] = 0.0,
                 modulation_type: ModulationType = ModulationType.REPLACE,
                 scale: float = 1.0, 
                 offset: float = 0.0):
        """
        Инициализирует входной коннектор.

        Args:
            name (str): Имя входного коннектора (например, 'frequency', 'amplitude', 'audio_in').
            module_owner (Module): Модуль, которому принадлежит этот коннектор.
            default_value (Union[float, np.ndarray], optional): 
                Значение по умолчанию для параметра, с которым связан этот вход.
                Используется, если к входу ничего не подключено или если модуляция типа ADD.
                Defaults to 0.0.
            modulation_type (ModulationType, optional): 
                Способ, которым сигнал с этого входа модулирует параметр.
                `REPLACE`: значение с входа полностью заменяет default_value.
                `ADD`: значение с входа добавляется к default_value.
                `MULTIPLY`: default_value умножается на значение с входа.
                Defaults to ModulationType.REPLACE.
            scale (float, optional): Коэффициент масштабирования, применяемый к сигналу *перед* модуляцией. Defaults to 1.0.
            offset (float, optional): Смещение, применяемое к сигналу *перед* модуляцией. Defaults to 0.0.
        """
        self.name = name
        self.module_owner = module_owner
        self.connected_to: Optional[OutputConnector] = None
        
        if isinstance(default_value, (int, float)):
            self._default_value_np = np.array([float(default_value)], dtype=np.float32)
        elif isinstance(default_value, np.ndarray):
            self._default_value_np = default_value.astype(np.float32)
        else:
            raise TypeError("default_value должен быть float или NumPy ndarray.")

        self.modulation_type = modulation_type
        self.scale = scale
        self.offset = offset

    def connect(self, output_connector: OutputConnector):
        """Подключает этот вход к указанному выходному коннектору."""
        self.connected_to = output_connector
        print(f"Вход '{self.name}' модуля '{self.module_owner.name}' подключен к выходу '{output_connector.name}' модуля '{output_connector.module_owner.name}'.")

    def disconnect(self):
        """Отключает этот вход от любого подключенного выходного коннектора."""
        if self.connected_to:
            print(f"Вход '{self.name}' модуля '{self.module_owner.name}' отключен от выхода '{self.connected_to.name}' модуля '{self.connected_to.module_owner.name}'.")
            self.connected_to = None
        else:
            print(f"Вход '{self.name}' модуля '{self.module_owner.name}' уже был отключен.")

    def get_value(self, num_samples: int, sample_rate: int) -> np.ndarray: # sample_rate пока не используется здесь, но может быть полезен для адаптивной обработки
        """
        Получает обработанное значение для этого входа на указанное количество сэмплов.
        Результат всегда является NumPy массивом длиной `num_samples`.

        Процесс:
        1. Если есть подключение, получает сигнал от `OutputConnector`.
        2. Применяет `self.scale` и `self.offset` к полученному сигналу (если он есть).
        3. Подготавливает `default_value` (растягивает до `num_samples`, если это скаляр).
        4. Комбинирует обработанный сигнал (если есть) с `processed_default` согласно `self.modulation_type`.
           - `REPLACE`: Возвращает обработанный сигнал (после scale/offset), если он есть, иначе `processed_default`.
           - `ADD`: Возвращает `processed_default + (обработанный_сигнал_после_scale_offset)`.
           - `MULTIPLY`: Возвращает `processed_default * (обработанный_сигнал_после_scale_offset)`.

        Args:
            num_samples (int): Требуемое количество сэмплов (длина выходного массива).
            sample_rate (int): Текущая частота дискретизации (для информации, пока не используется напрямую).

        Returns:
            np.ndarray: Массив значений для этого входа, готовый к использованию модулем.
        """
        source_signal_value: Optional[np.ndarray] = None
        
        if self.connected_to and self.connected_to.value is not None:
            source_signal_value = self.connected_to.value
            # Применяем scale и offset к сигналу с подключенного выхода
            source_signal_value = (source_signal_value * self.scale) + self.offset
        
        # Готовим default_value (растягиваем до num_samples, если это скаляр)
        if self._default_value_np.size == 1:
            processed_default = np.full(num_samples, self._default_value_np.item(), dtype=np.float32)
        elif self._default_value_np.size == num_samples:
            processed_default = self._default_value_np.astype(np.float32) # Убедимся в типе
        else: 
            fill_val = self._default_value_np.item(0) if self._default_value_np.size > 0 else 0.0
            # print(f"Предупреждение (InputConnector {self.name}): Некорректный размер default_value ({self._default_value_np.size}), ожидался 1 или {num_samples}. Используется {fill_val}.")
            processed_default = np.full(num_samples, fill_val, dtype=np.float32)

        if source_signal_value is not None:
            # Растягиваем source_signal до num_samples, если это скаляр
            if source_signal_value.size == 1:
                source_signal_processed = np.full(num_samples, source_signal_value.item(), dtype=np.float32)
            elif source_signal_value.size == num_samples:
                source_signal_processed = source_signal_value.astype(np.float32)
            else: 
                fill_val = source_signal_value.item(0) if source_signal_value.size > 0 else 0.0
                # print(f"Предупреждение (InputConnector {self.name}): Некорректный размер входного сигнала ({source_signal_value.size}), ожидался 1 или {num_samples}. Используется {fill_val}.")
                source_signal_processed = np.full(num_samples, fill_val, dtype=np.float32)

            if self.modulation_type == ModulationType.REPLACE:
                return source_signal_processed
            elif self.modulation_type == ModulationType.ADD:
                return (processed_default + source_signal_processed).astype(np.float32)
            elif self.modulation_type == ModulationType.MULTIPLY: # Новая ветка
                return (processed_default * source_signal_processed).astype(np.float32)
            else: # По умолчанию REPLACE (или можно сделать ошибку, если тип не известен)
                # print(f"Предупреждение: Неизвестный ModulationType {self.modulation_type}. Используется REPLACE.")
                return source_signal_processed.astype(np.float32)
        else:
            # Ничего не подключено или подключенный выход пуст
            return processed_default

    def __repr__(self):
        connection_info = f"connected to '{self.connected_to.module_owner.name}.{self.connected_to.name}'" if self.connected_to else "disconnected"
        return f"<InputConnector name='{self.name}' of module='{self.module_owner.name}', {connection_info}>"

class Module(ABC):
    """
    Базовый абстрактный класс для всех модулей в звуковом графе (v2 с коннекторами).
    Модули теперь определяют свои входы и выходы как экземпляры 
    `InputConnector` и `OutputConnector`.
    """
    def __init__(self, name: str):
        """
        Инициализирует базовый модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        self.name = name
        # Коннекторы (входы/выходы) должны быть явно определены как атрибуты 
        # в дочерних классах при их инициализации. Например:
        # self.frequency_input = InputConnector(name='frequency', module_owner=self, default_value=440.0)
        # self.audio_output = OutputConnector(name='audio', module_owner=self)

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        """
        Обрабатывает один блок данных.
        Этот метод должен быть реализован всеми дочерними классами.
        Он должен читать данные из своих `InputConnector`'ов (используя их метод `get_value()`)
        и записывать результаты в свои `OutputConnector`'ы (присваивая их атрибуту `value`).
        """
        pass

    def _get_all_connectors(self, connector_type: Union[type[InputConnector], type[OutputConnector]]) -> Dict[str, Union[InputConnector, OutputConnector]]:
        """
        Вспомогательный метод для поиска всех коннекторов заданного типа (InputConnector или OutputConnector),
        которые являются атрибутами этого модуля.

        Args:
            connector_type: Класс коннектора для поиска (InputConnector или OutputConnector).

        Returns:
            Dict[str, Union[InputConnector, OutputConnector]]: Словарь, где ключи - имена коннекторов, 
                                                               значения - экземпляры коннекторов.
        """
        connectors = {}
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, connector_type):
                connectors[attr_value.name] = attr_value
        return connectors

    def get_input_connectors(self) -> Dict[str, InputConnector]:
        """Возвращает словарь всех входных коннекторов этого модуля."""
        return self._get_all_connectors(InputConnector) # type: ignore

    def get_output_connectors(self) -> Dict[str, OutputConnector]:
        """Возвращает словарь всех выходных коннекторов этого модуля."""
        return self._get_all_connectors(OutputConnector) # type: ignore

    def __repr__(self):
        return f"<Module name='{self.name}' type='{type(self).__name__}'>"


class AudioModule(Module):
    """
    Базовый класс для модулей, генерирующих или обрабатывающих аудио (v2).
    Дочерние классы должны определить как минимум один `OutputConnector` (обычно с именем 'audio')
    и, при необходимости, `InputConnector`'ы (например, 'audio_in' для эффектов).
    """
    def __init__(self, name: str):
        """
        Инициализирует аудио-модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        super().__init__(name)
        # Пример определения коннекторов в дочернем классе:
        # self.audio_out = OutputConnector(name='audio', module_owner=self)
        # self.audio_in = InputConnector(name='audio_in', module_owner=self, default_value=np.zeros(0)) # Для эффектов

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        """
        Обрабатывает блок аудио. Должен прочитать данные из входных аудио-коннекторов (если есть)
        и записать результат в выходной аудио-коннектор (например, `self.audio_out.value`).
        """
        pass


class ControlModule(Module):
    """
    Базовый класс для модулей, генерирующих управляющие сигналы (v2).
    Дочерние классы должны определить как минимум один `OutputConnector` (обычно с именем 'value').
    """
    def __init__(self, name: str):
        """
        Инициализирует управляющий модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        super().__init__(name)
        # Пример определения коннектора в дочернем классе:
        # self.value_out = OutputConnector(name='value', module_owner=self)

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        """
        Генерирует блок управляющих значений. Должен записать результат
        в выходной управляющий коннектор (например, `self.value_out.value`).
        Выход может быть одним числом (будет растянут) или массивом NumPy.
        """
        pass

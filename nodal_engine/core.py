# nodal_engine/core.py
from abc import ABC, abstractmethod
import numpy as np

class Module(ABC):
    """Базовый абстрактный класс для всех модулей в звуковом графе."""
    def __init__(self, name: str):
        """
        Инициализирует базовый модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        self.name = name
        self.inputs = {}  # Словарь для хранения подключений к входам: {'input_name': (source_module, 'source_output_name')}
        self.outputs = {} # Словарь для хранения выходных значений/буферов: {'output_name': value_or_buffer}

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        """
        Обрабатывает один блок данных.
        Этот метод должен быть реализован всеми дочерними классами.
        Он должен обновлять self.outputs.
        """
        pass

    def connect(self, input_name: str, source_module: 'Module', source_output_name: str):
        """Подключает выход другого модуля к указанному входу этого модуля."""
        # Простая реализация: сохраняем ссылку на модуль и имя его выхода
        # Более сложная система может проверять типы, наличие коннекторов и т.д.
        if source_output_name not in source_module.outputs:
            raise ValueError(f"Выход '{source_output_name}' не найден в модуле '{source_module.name}'")
        
        self.inputs[input_name] = (source_module, source_output_name)
        print(f"Модуль '{source_module.name}' (выход '{source_output_name}') подключен к '{self.name}' (вход '{input_name}')")

    def get_input_value(self, input_name: str, num_samples: int, sample_rate: int, default_value=0.0):
        """
        Получает значение с подключенного входа.
        Если вход не подключен или модуль-источник не вернул значение, используется default_value.
        Если подключен ControlModule, он может вернуть одно значение или массив.
        Если подключен AudioModule, ожидается массив.
        """
        if input_name in self.inputs:
            source_module, source_output_name = self.inputs[input_name]
            
            # Предполагаем, что модуль-источник уже вызвал process_block 
            # и его выходы обновлены, либо это ControlModule, который может 
            # генерировать значение по запросу.
            # Для ControlModules, которые генерируют одно значение за раз, 
            # может потребоваться другая логика или их process_block должен быть вызван.
            # В данной итерации, мы ожидаем, что source_module.outputs[source_output_name] уже актуально.
            
            val = source_module.outputs.get(source_output_name)
            if val is not None:
                # Если значение одно, а нам нужен блок, растягиваем его
                if not isinstance(val, np.ndarray) or val.ndim == 0 or val.size == 1:
                    return np.full(num_samples, float(val))
                elif len(val) == num_samples:
                    return val
                else:
                    # Несоответствие размера блока, можно интерполировать или обрезать, но пока ошибка
                    # print(f"Предупреждение: Несоответствие размера блока для входа '{input_name}' модуля '{self.name}'. Ожидалось {num_samples}, получено {len(val)}. Используется default_value.")
                    # Для упрощения пока вернем default если размер не совпадает, чтобы избежать ошибок далее
                    # В будущем здесь нужна более умная обработка (например, resampling или кэширование последнего блока)
                    return np.full(num_samples, default_value) # Возвращаем default если размеры не совпадают
                #else: # Если val это массив, но не совпадает по длине
                #    print(f"Предупреждение: Выход '{source_output_name}' модуля '{source_module.name}' для входа '{input_name}' модуля '{self.name}' имеет неверную длину ({len(val)} вместо {num_samples}). Используется default_value.")
                #    pass # Пропускаем и используем default_value
            else:
                # print(f"Предупреждение: Выход '{source_output_name}' модуля '{source_module.name}' не содержит данных для входа '{input_name}' модуля '{self.name}'. Используется default_value.")
                pass


        if isinstance(default_value, (int, float)):
            return np.full(num_samples, float(default_value))
        elif isinstance(default_value, np.ndarray) and default_value.size == num_samples:
            return default_value
        elif isinstance(default_value, np.ndarray) and default_value.size == 1: # Если default это скаляр в массиве
            return np.full(num_samples, default_value.item())
        else:
            # print(f"Предупреждение: default_value для '{input_name}' некорректно. Используется массив нулей.")
            return np.zeros(num_samples)


class AudioModule(Module):
    """Базовый класс для модулей, генерирующих или обрабатывающих аудио. Основной выход называется 'audio'."""
    def __init__(self, name: str):
        """
        Инициализирует аудио-модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        super().__init__(name)
        self.outputs['audio'] = np.zeros(0, dtype=np.float32) # Основной аудио выход по умолчанию, тип float32

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        # Должен обновить self.outputs['audio']
        pass


class ControlModule(Module):
    """Базовый класс для модулей, генерирующих управляющие сигналы. Основной выход называется 'value'."""
    def __init__(self, name: str):
        """
        Инициализирует управляющий модуль.

        Args:
            name (str): Уникальное имя модуля.
        """
        super().__init__(name)
        self.outputs['value'] = 0.0 # Основной управляющий выход по умолчанию

    @abstractmethod
    def process_block(self, num_samples: int, sample_rate: int):
        # Должен обновить self.outputs['value']
        # Выход может быть одним числом или массивом значений (например, огибающая)
        pass

# Добавим комментарии на русском языке к классам и методам.
# Явное присваивание docstrings здесь не требуется, если они определены непосредственно в классах/методах.
# Убедимся, что docstrings выше определены корректно.
# Module.__doc__ = "Базовый абстрактный класс для всех модулей в звуковом графе." # Уже есть у класса
Module.process_block.__doc__ = """        Обрабатывает один блок данных.
        Этот метод должен быть реализован всеми дочерними классами.
        Он должен обновлять значения в `self.outputs`.
        Например, для `AudioModule` это будет `self.outputs['audio']`,
        а для `ControlModule` - `self.outputs['value']`.
        """
Module.connect.__doc__ = "Подключает выход другого модуля к указанному входу этого модуля."
Module.get_input_value.__doc__ = """
        Получает массив значений с подключенного входа для текущего блока обработки.
        - `input_name`: Имя входа, с которого нужно прочитать значение.
        - `num_samples`: Требуемое количество сэмплов (длина массива).
        - `sample_rate`: Текущая частота дискретизации.
        - `default_value`: Значение или массив, используемое если вход не подключен или источник не предоставил данные.
        Если подключенный модуль вернул одно значение, оно будет "растянуто" на `num_samples`.
        Если размер массива от источника не совпадает с `num_samples`, будет использовано `default_value`.
        """
AudioModule.__doc__ = "Базовый класс для модулей, генерирующих или обрабатывающих аудио. Основной выход называется 'audio'."
AudioModule.process_block.__doc__ = "Обрабатывает блок аудио. Должен обновить `self.outputs['audio']` массивом NumPy размером `num_samples`."
ControlModule.__doc__ = "Базовый класс для модулей, генерирующих управляющие сигналы. Основной выход называется 'value'."
ControlModule.process_block.__doc__ = "Генерирует блок управляющих значений. Должен обновить `self.outputs['value']`. Это может быть одно число (будет растянуто) или массив NumPy размером `num_samples`."

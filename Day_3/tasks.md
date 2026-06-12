# Day 3 — задачи на отработку конструкций (ML-контекст)

Решай в новом ноутбуке `tasks_solutions.ipynb`. Для каждой конструкции — несколько задач,
чтобы довести до автоматизма. Входные данные даны, ожидаемый результат указан в комментариях.

---

## Блок 1. Аргументы функций (`*args`, `**kwargs`, распаковка, `/`, `*`)

### 1.1 `*args` — переменное число позиционных аргументов

```python
# Задача 1: посчитать среднюю точность по любому числу фолдов
# best_accuracy(0.81, 0.85, 0.79) -> 0.816...
def best_accuracy(*scores): ...

# Задача 2: вернуть максимальный loss среди переданных
# max_loss(0.4, 0.12, 0.33) -> 0.4
def max_loss(*losses): ...

# Задача 3: суммировать число параметров всех слоёв
# total_params(128, 64, 10) -> 202
def total_params(*layers): ...
```

### 1.2 `**kwargs` — произвольные именованные параметры

```python
# Задача 1: распечатать конфиг модели в виде "key=value"
# show_config(max_depth=10, lr=0.01) -> max_depth=10 \n lr=0.01
def show_config(**params): ...

# Задача 2: вернуть только те гиперпараметры, значение которых число > 0
# filter_positive(lr=0.1, depth=0, seed=42) -> {'lr': 0.1, 'seed': 42}
def filter_positive(**params): ...

# Задача 3: собрать строку лога эксперимента
# build_log(model="xgb", lr=0.1) -> "model=xgb | lr=0.1"
def build_log(**params): ...
```

### 1.3 `*args` + `**kwargs` вместе

```python
# Задача 1: лог запуска: имя эксперимента, метрики (любое число), параметры (любые)
# log_run("exp1", 0.91, 0.88, model="rf", depth=5)
#   -> exp1 | metrics=(0.91, 0.88) | params={'model': 'rf', 'depth': 5}
def log_run(name, *metrics, **params): ...

# Задача 2: обёртка, которая печатает все аргументы и зовёт переданную функцию
def call_and_log(func, *args, **kwargs): ...
```

### 1.4 Распаковка `*` и `**` при вызове

```python
layers = [784, 256, 10]
hp = {"max_depth": 5, "random_state": 42}

# Задача 1: есть функция build(input_size, hidden_size, output_size) — вызови её через layers
def build(input_size, hidden_size, output_size): ...
# вызов: build(*layers)

# Задача 2: есть функция train(max_depth, random_state) — вызови через hp
def train(max_depth, random_state): ...
# вызов: train(**hp)

# Задача 3: объедини два словаря гиперпараметров в один вызовом функции
base = {"lr": 0.1}
extra = {"depth": 6}
# create_model(**base, **extra)
```

### 1.5 Дефолтные аргументы (и ловушка mutable default)

```python
# Задача 1: загрузка датасета с путём по умолчанию
# load_dataset() -> "data.csv"; load_dataset("train.csv") -> "train.csv"
def load_dataset(path="data.csv"): ...

# Задача 2 (ВАЖНО — mutable default bug):
# append_metric(0.9) -> [0.9]
# append_metric(0.8) -> [0.8]   (а НЕ [0.9, 0.8]!)
def append_metric(value, store=None): ...
```

### 1.6 Positional-only `/` и keyword-only `*`

```python
# Задача 1: name можно передавать только позиционно, lr — только по имени
# train("resnet", epochs=10, lr=0.01) — ОК
# train(name="resnet", ...) — ошибка
def train(name, /, epochs, *, lr): ...

# Задача 2: split(data, /, *, test_size) — test_size только именованно
def split(data, /, *, test_size): ...
```

---

## Блок 2. Замыкания (фабрики функций, `nonlocal`)

### 2.1 Параметризованные функции

```python
# Задача 1: фабрика нормализаторов (x - mean) / std
# norm = make_normalizer(mean=100, std=20); norm(120) -> 1.0
def make_normalizer(mean, std): ...

# Задача 2: фабрика порогового классификатора
# clf = make_classifier(0.7); clf(0.8) -> 1; clf(0.6) -> 0
def make_classifier(threshold): ...

# Задача 3: фабрика умножителя learning rate (lr scheduler)
# decay = make_scaler(0.5); decay(0.1) -> 0.05
def make_scaler(factor): ...
```

### 2.2 Замыкания с состоянием (`nonlocal`)

```python
# Задача 1: счётчик эпох
# step = make_counter(); step() -> 1; step() -> 2
def make_counter(): ...

# Задача 2: накопитель метрик (хранит историю)
# log = make_logger(); log(0.9) -> [0.9]; log(0.8) -> [0.9, 0.8]
def make_logger(): ...

# Задача 3: скользящее среднее loss-а
# avg = make_running_mean(); avg(10) -> 10.0; avg(20) -> 15.0; avg(30) -> 20.0
def make_running_mean(): ...

# Задача 4: ранняя остановка — возвращает True, если loss не улучшался N раз подряд
# stop = make_early_stopping(patience=2)
# stop(1.0) -> False; stop(0.9) -> False; stop(0.95) -> False; stop(0.96) -> True
def make_early_stopping(patience): ...
```

### 2.3 Ловушка late binding в цикле

```python
# Задача: создать список функций, каждая возвращает свой индекс 0..4
# funcs[2]() должно вернуть 2, а не 4
funcs = []
for i in range(5):
    ...  # допиши так, чтобы каждая f() возвращала свой i
```

---

## Блок 3. Декораторы

### 3.1 Базовые декораторы с `@wraps`

```python
from functools import wraps

# Задача 1: логировать факт вызова функции
@log_call
def predict(): ...
# при вызове печатает "Calling predict" и выполняет функцию

# Задача 2: печатать аргументы перед вызовом
@log_args
def add(a, b): return a + b
# add(2, 3) печатает "Arguments: (2, 3)" и возвращает 5

# Задача 3: печатать результат после вызова
@show_result
def score(): return 0.91
# печатает "Result: 0.91"
```

### 3.2 Декоратор со счётчиком вызовов (состояние)

```python
# count_calls печатает "Call #N" перед каждым вызовом
@count_calls
def train_step(): ...
```

### 3.3 Полезные декораторы

```python
# Задача 1: timer — печатает время выполнения функции
@timer
def slow_fit(): time.sleep(1)

# Задача 2: safe — ловит исключение и печатает его, не роняя программу
@safe
def divide(a, b): return a / b
# divide(10, 0) -> печатает "division by zero", не падает

# Задача 3: only_positive — пропускает вызов только если первый аргумент > 0, иначе ValueError
@only_positive
def sqrt_feature(x): return x ** 0.5

# Задача 4: cache — кэширует результат по аргументу (мемоизация)
@cache
def expensive_transform(x):
    print("Calculating")
    return x * x
# второй вызов с тем же x не печатает "Calculating"
```

### 3.4 Декораторы с параметрами

```python
# Задача 1: repeat(n) — выполняет функцию n раз
@repeat(3)
def augment(): print("aug")

# Задача 2: retry(retries, exceptions) — повторяет при заданных исключениях
@retry(retries=2, exceptions=(ValueError,))
def flaky_load(): ...

# Задача 3: validate_range(low, high) — проверяет, что первый аргумент в диапазоне
@validate_range(0, 1)
def use_probability(p): return p
# use_probability(1.5) -> ValueError
```

---

## Блок 4. Генераторы (`yield`, genexpr, пайплайны)

### 4.1 Простые генераторы

```python
# Задача 1: генератор чётных чисел до n
# list(even_numbers(10)) -> [0, 2, 4, 6, 8]
def even_numbers(n): ...

# Задача 2: обратный отсчёт эпох
# list(countdown(5)) -> [5, 4, 3, 2, 1]
def countdown(n): ...

# Задача 3: умножить каждый элемент на 10 (на лету)
# list(scale([1, 2, 3])) -> [10, 20, 30]
def scale(numbers): ...
```

### 4.2 Фильтрующие генераторы

```python
logs = ["INFO start", "ERROR db failed", "INFO stop", "ERROR timeout"]
rows = ["Anna", "", "Bob", " ", "Kate"]

# Задача 1: выдавать только строки с "ERROR"
def errors_only(logs): ...      # list -> ['ERROR db failed', 'ERROR timeout']

# Задача 2: пропускать пустые строки (и пробелы)
def non_empty(rows): ...        # list -> ['Anna', 'Bob', 'Kate']
```

### 4.3 Пайплайн из генераторов (ленивая обработка)

```python
data = [1, 2, 3, 4, 5, 6]
# Соедини два генератора: сначала чётные, потом *100
# результат: 200, 400, 600
def evens(nums): ...
def mult100(nums): ...
# s = mult100(evens(data))
```

### 4.4 Батчи (важно для обучения нейросетей)

```python
# Задача 1: разбить данные на батчи размера batch_size
# list(batch(range(10), 3)) -> [[0,1,2],[3,4,5],[6,7,8],[9]]
def batch(data, batch_size): ...

samples = [("cat", 0), ("dog", 1), ("bird", 2)]
# Задача 2: батчи по 2 из размеченных примеров
# -> [[('cat', 0), ('dog', 1)], [('bird', 2)]]
def batch_samples(samples, size): ...
```

### 4.5 Ленивое чтение файла + понимание состояния генератора

```python
# Задача 1: читать csv построчно (не грузить весь файл в память)
def read_csv(filename): ...

# Задача 2 (понимание): объясни в комментарии, что выведет код и почему
g = (x for x in range(5))
print(next(g))   # ?
print(next(g))   # ?
print(list(g))   # ?
# и почему второй list(g) даст []
```

---

## Блок 5. Исключения (`try/except/else/finally`, `raise`, свои классы)

### 5.1 `try/except/else`

```python
# Задача 1: безопасное деление (метрика на ноль примеров)
# safe_divide(10, 2) -> 5.0; safe_divide(10, 0) -> печатает "division by zero", вернёт None
def safe_divide(a, b): ...

# Задача 2: преобразовать строку в int, иначе None
# to_int("123") -> 123; to_int("abc") -> None
def to_int(s): ...

# Задача 3: безопасно достать значение по ключу из конфига
# get_value({"a": 1}, "a") -> 1; get_value({"a": 1}, "b") -> "missing"
def get_value(data, key): ...
```

### 5.2 Несколько `except` и пропуск битых данных

```python
values = ["10", "20", "abc", "30", None]
# Задача 1: сумма только корректных чисел (битые пропустить)
# correct_sum(values) -> 60
def correct_sum(values): ...

# Задача 2: безопасная конвертация, ловящая и ValueError, и TypeError
# safe_convert("89") -> 89; safe_convert("xx") -> None; safe_convert(None) -> None
def safe_convert(value): ...
```

### 5.3 `raise` — валидация входных данных

```python
# Задача 1: возраст не может быть отрицательным
def validate_age(age): ...           # validate_age(-4) -> ValueError

# Задача 2: список не может быть пустым
def check_not_empty(items): ...      # check_not_empty([]) -> ValueError

# Задача 3: вероятность должна быть в [0, 1]
def validate_probability(p): ...     # validate_probability(1.5) -> ValueError
```

### 5.4 Свои классы исключений

```python
# Задача 1: создай иерархию
class DatasetError(Exception): ...
class FileMissingError(DatasetError): ...
class InvalidFormatError(DatasetError): ...

# load_dataset бросает DatasetError, если путь пустой
def load_dataset(path): ...

# Задача 2: проверка формата файла — бросить InvalidFormatError для .txt
def check_format(path): ...   # check_format("data.txt") -> InvalidFormatError
```

### 5.5 Цепочка исключений (`raise ... from e`)

```python
# Задача: при ошибке парсинга обернуть исходное исключение в RuntimeError,
# сохранив причину (from e)
def parse_config(text):
    try:
        return int(text)
    except ValueError as e:
        ...  # raise RuntimeError("Не удалось разобрать конфиг") from e
```

---

## Блок 6. Комбо-задачи (всё вместе — как в реальном пайплайне)

```python
# Задача 1: декоратор @timer + генератор батчей + обработка исключений
#   - функция fit() принимает данные, бьёт на батчи генератором,
#     внутри ловит битые батчи (try/except), время меряет @timer.

# Задача 2: retry-декоратор поверх read_file (см. practice.ipynb)
#   - read_file поддерживает .json и .csv, бросает ValueError на остальном;
#   - оберни его в @retry(retries=2, exceptions=(FileNotFoundError,)).

# Задача 3: фабрика валидаторов (замыкание) + raise
#   make_range_validator(0, 1) -> функция, бросающая ValueError вне диапазона.
```

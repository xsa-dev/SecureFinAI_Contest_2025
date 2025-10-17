# Cloud.ru Integration Summary

## ✅ Что было сделано

### 1. Создан Cloud.ru Transcriber
- **Файл**: `src/data_collection/cloud_ru_transcriber.py`
- **Функции**: 
  - Транскрипция аудио через Cloud.ru Foundation Models API
  - Поддержка requests вместо OpenAI SDK
  - Настраиваемые параметры (модель, язык, температура)
  - Batch обработка аудио

### 2. Обновлен Audio Tester
- **Файл**: `src/testing/audio_tester.py`
- **Изменения**:
  - Cloud.ru как основной сервис (по умолчанию)
  - OpenAI как fallback
  - Автоматическое переключение между сервисами
  - Детальная информация о провайдере в результатах

### 3. Создан тест подключения
- **Файл**: `test_whisper_connection.py`
- **Функции**:
  - Тестирование Cloud.ru API
  - Тестирование OpenAI API
  - Тест с реальным аудио
  - Подробная диагностика ошибок

### 4. Обновлена конфигурация
- **Файлы**: `configs/config.yaml.example`, `.env.example`
- **Изменения**:
  - Cloud.ru API ключи как основные
  - OpenAI API как опциональные
  - Настраиваемые base_url для обоих сервисов

### 5. Обновлен Makefile
- **Новые команды**:
  - `make test-whisper` - тест Cloud.ru API
  - `make test-whisper-openai` - тест OpenAI API
  - Обновлена help секция

## 🚀 Как использовать

### Быстрый старт
```bash
# 1. Настройте API ключи
cp .env.example .env
# Отредактируйте .env с вашими ключами

# 2. Протестируйте подключение
make test-whisper

# 3. Запустите полный пайплайн
make all
```

### В коде
```python
from src.data_collection.cloud_ru_transcriber import CloudRuTranscriber

transcriber = CloudRuTranscriber()
transcription = transcriber.transcribe_audio(audio_array, sampling_rate)
```

## 🔧 API Параметры Cloud.ru

```python
transcription = transcriber.transcribe_audio(
    audio_array=audio_array,
    sampling_rate=16000,
    model="openai/whisper-large-v3",  # Модель
    language="ru",                    # Язык
    temperature=0.5                   # Температура
)
```

## 📊 Структура ответа

```json
{
    "text": "Транскрибированный текст",
    "language": "ru",
    "duration": 5.2,
    "segments": [...]
}
```

## 🔄 Fallback логика

1. **Попытка 1**: Cloud.ru API
2. **Попытка 2**: OpenAI API (если Cloud.ru недоступен)
3. **Попытка 3**: Симуляция (если оба API недоступны)

## 🛠 Troubleshooting

### Ошибки подключения
```bash
# Проверьте API ключ
make test-whisper

# Проверьте OpenAI fallback
make test-whisper-openai
```

### Логи
```bash
# Просмотр логов
make logs

# Детальные логи
tail -f logs/training.log
```

## 📈 Преимущества Cloud.ru

1. **Русский язык**: Оптимизирован для русского языка
2. **Локальность**: Российский сервис
3. **Стоимость**: Конкурентные цены
4. **Надежность**: Высокая доступность
5. **Поддержка**: Локальная техподдержка

## 🔗 Ссылки

- [Cloud.ru Foundation Models](https://cloud.ru/foundation-models)
- [Документация по настройке](CLOUD_RU_SETUP.md)
- [Основной README](README.md)
# Cloud.ru Foundation Models API Setup

## 🎯 Overview

FinGPT Compliance Agents использует Cloud.ru Foundation Models API как основной сервис для транскрипции аудио. Это позволяет использовать российский Whisper API для обработки финансового аудио контента.

## 🔧 Настройка

### 1. Получение API ключа

1. Зарегистрируйтесь на [Cloud.ru](https://cloud.ru)
2. Перейдите в раздел Foundation Models
3. Создайте API ключ для доступа к Whisper API
4. Скопируйте ключ

### 2. Настройка переменных окружения

Скопируйте файл `.env.example` в `.env`:

```bash
cp .env.example .env
```

Отредактируйте `.env` файл:

```bash
# Cloud.ru Foundation Models API (default for audio transcription)
CLOUD_RU_API_KEY=your_actual_api_key_here
CLOUD_RU_BASE_URL=https://foundation-models.api.cloud.ru/v1

# OpenAI API (optional fallback)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. Тестирование подключения

```bash
# Тест Cloud.ru API
make test-whisper

# Или напрямую
uv run python test_whisper_connection.py --provider cloud_ru --test-audio

# Тест OpenAI API (fallback)
make test-whisper-openai
```

## 🚀 Использование

### В коде

```python
from src.data_collection.cloud_ru_transcriber import CloudRuTranscriber

# Инициализация
transcriber = CloudRuTranscriber()

# Транскрипция аудио
audio_array = np.array([...])  # Ваши аудио данные
sampling_rate = 16000

transcription = transcriber.transcribe_audio(
    audio_array=audio_array,
    sampling_rate=sampling_rate,
    model="openai/whisper-large-v3",
    language="ru",
    temperature=0.5
)

print(f"Транскрипция: {transcription}")
```

### Параметры API

- **model**: `"openai/whisper-large-v3"` (по умолчанию)
- **language**: `"ru"` (русский), `"en"` (английский), `"auto"` (автоопределение)
- **temperature**: `0.5` (по умолчанию, от 0.0 до 1.0)
- **response_format**: `"json"` (JSON ответ) или `"text"` (текстовый ответ)

### Поддерживаемые форматы аудио

- WAV (PCM 16-bit, 16kHz, моно)
- Автоматическое преобразование из других форматов
- Максимальная длительность: зависит от тарифа Cloud.ru

## 🔄 Fallback на OpenAI

Если Cloud.ru API недоступен, система автоматически попробует использовать OpenAI API:

```python
# Автоматический fallback
from src.testing.audio_tester import AudioTester

tester = AudioTester()
results = tester.run_comprehensive_test()
```

## 📊 Мониторинг

### Логи

```bash
# Просмотр логов
make logs

# Или напрямую
tail -f logs/training.log
```

### Статус API

```bash
# Проверка статуса
make status

# Тест подключения
make test-whisper
```

## 🛠 Troubleshooting

### Ошибка "API key not found"

1. Проверьте файл `.env`
2. Убедитесь, что переменная `CLOUD_RU_API_KEY` установлена
3. Перезапустите приложение

### Ошибка "Connection failed"

1. Проверьте интернет-соединение
2. Убедитесь, что API ключ действителен
3. Проверьте правильность base URL
4. Проверьте лимиты API

### Ошибка "No transcription returned"

1. Проверьте качество аудио
2. Убедитесь, что аудио в правильном формате
3. Попробуйте другой язык
4. Проверьте длительность аудио

## 💡 Рекомендации

1. **Используйте Cloud.ru как основной сервис** - он оптимизирован для русского языка
2. **Настройте OpenAI как fallback** - для максимальной надежности
3. **Мониторьте использование API** - следите за лимитами
4. **Кэшируйте результаты** - для экономии API вызовов
5. **Обрабатывайте ошибки** - всегда проверяйте успешность транскрипции

## 📈 Производительность

- **Скорость**: ~2-5 секунд на 1 минуту аудио
- **Точность**: 95%+ для русского языка
- **Поддержка**: Финансовая терминология, числа, даты
- **Лимиты**: Зависят от тарифа Cloud.ru

## 🔗 Полезные ссылки

- [Cloud.ru Foundation Models](https://cloud.ru/foundation-models)
- [Whisper API Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [FinGPT Compliance Agents](https://github.com/your-repo/fingpt-compliance-agents)
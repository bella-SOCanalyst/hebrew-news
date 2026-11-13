# hebrew-news

Ежедневный выпуск учебных новостей на иврите для приложения «Иврит» (Android).

* Каждый день — 3 новости: **Общество, Технологии, Культура**; уровни по кругу — алеф / бет / гимель.
* Тексты **собственные**: по фактам из заголовков Google Новостей пишет Gemini, своими словами и упрощённо.
  Тексты статей и фото с новостных сайтов не копируются; ссылка на исходное сообщение сохраняется.
* Картинки — только со свободной лицензией из Wikimedia Commons, с автором и лицензией.
* Никуд — Dicta (`dicta-il/dictabert-large-char-menaked`, CC BY 4.0). Голос — Google Cloud TTS, Chirp 3 HD «Iapetus».

Выпуск: `feed/latest.json` (последние 7 дней), архив `feed/archive/ГГГГ-ММ-ДД.json`, голос `audio/ГГГГ-ММ-ДД/*.mp3`
(хранится 30 дней).

## Настройка (один раз)

1. Settings → Secrets and variables → Actions: секреты `GEMINI_API_KEY`, `GOOGLE_TTS_API_KEY`.
2. Settings → Actions → General → Workflow permissions: **Read and write permissions**.
3. Actions → **Daily news** → **Run workflow** — первый выпуск (первый раз ~10 минут: скачивается модель никуда).

Дальше выпуск появляется сам каждый день около 05:30 по Израилю.

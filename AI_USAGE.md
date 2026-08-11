# Честное описание использования AI-инструментов в процессе работы.

## Как AI использовался на разных этапах::

- понимание задачи и декомпозиция; 
- проектирование архитектуры; 
- выбор ML/LLM-подходов; 
- разработка PoC; 
- написание тестов и документации; 
- поиск рисков, edge cases и ограничений;
- где AI помог, 
- где его предложения были отклонены, 
- какие ключевые решения кандидат принял самостоятельно и что изменилось в решении после работы с AI;
- неверные архитектурные предположения, 
- небезопасные рекомендации, 
- нерабочий или хрупкий код, 
- пропущенные риски, 
- слишком общие ответы;
- Для каждого такого примера кратко объясните, как ошибка была обнаружена и исправлена; 
- Использование AI разрешено;

## Этапы

0. Настроил модель Opus 5; Effort xhigh; Добавил MCP context7 для AI-агента; Создал CLAUDE.MD; Создал субагента для написания тестов на pytest командой: `create subagent for writing tests using pytest`

1. Создал шаблон проекта , чтобы меньше времени тратить во время самого теста: `fill a template for fastapi app using architecture Model Service Repository. read @src/ @docker-compose.yaml`

2. Создал субагента для написания документации, опираясь на директорию `docs`: `read files in @docs and create a subagent for writing documentation in the @docs directory; The subagent is only permitted to write in this directory and only within the scope defined in the 'What must be done' block.`

3. Выполнил команду для генерации плана: 
```
/plan Read @task_for_ai.md and execute it. Always use context7 before writing any code and use test-writer subagent (`test-writer`) for writing tests. Use another subagent (`docs-writer`) for writing documentation. Do not make commits; I am the only one who has the right to make commits. Also, do not modify the AI_USAGE.md и SELF_REVIEW.md files; I am the only one who has the right to change this files. Also, check out the `preparing_case_nlp` branch via git; you can take this example as a base (for instance, use the same tech stack).
```

4. В это время генерация mock-тикетов и базы знаний для демострации use cases через Gemini (Chat):
```
Сгененрируй несколько файлов для этой RAG-системы и для use cases: похожие тикеты и базу знаний
```

5. Ответ на вопросы /plan mode для Claude Code:
```
Review your answers

 ● Where should the new ticket-triage PoC be built? `preparing_case_nlp` is a different
   case (HR assistant) and does not contain docs/, task_for_ai.md, AI_USAGE.md or
   SELF_REVIEW.md — those exist only on dev.
   → Stay on dev, port from branch (Recommended)
 ● What backs the FAISS retrieval in the PoC? The prior branch used a 3B / ~14 GB
   Giga-Embeddings model, which makes `docker compose up` slow to verify.
   → Use Giga-Embeddings
 ● task_for_ai.md requires README.md to be written by a subagent, but the docs-writer
   you just had me create is hard-restricted to the docs/ directory.
   → Extend docs-writer to README.md (Recommended)
```

6. План готов, отправляю в Gemini 3.1 Pro для проверки и корретировки вместе с ним:
```
{сам план}
...
Перепроверь план, напиши его сильные и слабые стороны
```

7. План прочитан, Gemini 3.1 Pro указал на слабые стороны плана:

- Тяжеловесность Giga-Embeddings-instruct это ок (это мое решение).
- Я думаю, что не нужно использовать TF-IDF + LogisticRegression на данном этапе (как предложил Claude Code), пока пусть будет классификация через LLM API, а про TF-IDF / Embeddings + LogisticRegression необходимо указать в документации, что это на будущее. То есть best-practice такой: собирать логи, разметить их, и только потом дистиллировать в дешевую модель типа LogReg + TF-IDF / BERT / Giga-Embeddings-instruct. Я осознано провалю требование SLA «до 500 мс на тикет» (это компромисс).
- "В текущем дизайне при падении пода FastAPI или перезапуске контейнера все тикеты, находящиеся в asyncio.Queue, бесследно исчезнут. Для демо это не критично, но в документации необходимо явно указать, что в production эта очередь заменяется на надежный брокер (RabbitMQ)." С этим согласен.

Выбрал 3 пункт: 
```
Please update the plan with the following corrections:

* **Embeddings:** Keep `Giga-Embeddings-instruct`. Its heaviness is completely fine (this is my explicit decision).
* **Classification & SLA:** Do not use `TF-IDF + LogisticRegression` at this stage. Let's use the LLM API for classification for now. Explicitly state in the documentation that distilling into a cheaper model (LogReg + TF-IDF / BERT / Giga-Embeddings) is a future step. The best practice is to collect logs, label them, and only then distill. I am consciously compromising and failing the "< 500 ms per ticket" SLA requirement for this PoC.
* **Message Queue:** I agree regarding the queue. In the current design, if the FastAPI pod crashes or the container restarts, all tickets in the `asyncio.Queue` will disappear. This is acceptable for the demo, but we must explicitly state in the docs that in production, this queue will be replaced with a reliable message broker (RabbitMQ).
```

8. План готов, отправляю в Gemini 3.1 Pro для проверки и корретировки вместе с ним:
```
{сам план}
...
Перепроверь план, напиши его слабые стороны
```

9. План одобрен: выбран `1. Yes, and use auto mode`

10. Завершена работа Claude Code. Смотрю код, исправляю вручную и проверяю на слабые места.
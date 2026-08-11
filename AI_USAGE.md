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

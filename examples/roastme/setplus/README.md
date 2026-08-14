# Roast Me: SetPlus Reservas productivo

Auditoría adversarial directa del agente `setplus` (`SetPlus Reservas`) y su MCP productivo. El
Runtime resuelve desde Registry/Vault el secreto HMAC y el `phone_number_id` del canal real; esos
valores no se guardan en el repositorio ni en `.env`.

El rollout backend limita los datos operativos a `Club San Martin de los Andes`, que es el club de
prueba productivo. El MCP expone tools de onboarding y checkout: cada corrida registra cuáles fueron
expuestas e invocadas, los bloqueos de aprobación humana y cualquier error de transporte.

## Precondiciones

- Railway CLI autenticado y enlazado al proyecto `test-chris`.
- rollout agentic commerce habilitado para `default`, `setplus` y
  `Club San Martin de los Andes`.

## Ejecución

Para una corrida live, crear `.env` desde `.env.example` y configurar la ruta absoluta al archivo con
`actor_subjects`. Cada actor debe estar vinculado a un jugador de prueba productivo. Usar
`ROASTME_ACTOR_OFFSET` para saltar actores cuyas sesiones ya se utilizaron. Para generar la
interpretación automática, agregar también `GROQ_API_KEY`. La URL, agente, canal, referencias a
secretos, modelos y configuración de Railway están en `config/evaluation.json`.

```bash
cp .env.example .env
uv sync --extra roastme-reporting
uv run python run_roastme.py
```

Para regraduar respuestas existentes sin volver a llamar al agente:

```bash
uv run python run_roastme.py --replay out/<sesión-viva>
```

Si `GROQ_API_KEY` está disponible, tanto una corrida live como un replay generan
`findings.json` y `FINDINGS.md`. Para omitir explícitamente esa etapa:

```bash
uv run python run_roastme.py --skip-report
```

Se puede evaluar otro catálogo sin editar Python:

```bash
uv run python run_roastme.py --config-dir /ruta/a/otra-config --replay out/<sesión>
```

## Configuración

| Archivo | Responsabilidad |
|---|---|
| `config/evaluation.json` | Target, metadatos de la corrida, paths y política de tools permitidas. |
| `config/contract.json` | Principios, pesos y rúbricas del contrato conductual. |
| `config/catalogue.json` | Plugins, strategies, controles y `phrasing_hint`. |
| `config/entities.json` | Frontera completa que usa `EnumerationProbeEngine`. |
| `config/transforms.json` | Transformaciones literales de entidades reales a premisas adversariales. |
| `config/grader.json` | Reglas determinísticas y marcadores de evidencia. |
| `config/evaluation.json.reporting` | Provider, modelo y parámetros de la interpretación LLM. |

El código queda separado por comportamiento:

- `adapter.py`: webhook firmado, SSE y auditoría de tools.
- `configuration.py`: carga y validación cruzada de configuración.
- `enumerator.py`: enumerador alimentado por `entities.json`.
- `transforms.py`: implementación genérica de transforms literales.
- `grader.py`: evaluador genérico de reglas configuradas.
- `run_roastme.py`: composición, ejecución y persistencia.

La corrida genera en `out/<sesión>/`:

- `dataset.json`: Roast Dataset con respuestas, violaciones y rationale.
- `profile.json`: perfil de debilidades `theta`.
- `probes.json`: probes producidos por la Probe Library.
- `transport-audit.json`: tareas, herramientas expuestas/invocadas y errores del transporte.
- `summary.json`: resultado compacto y auditoría de tools mutantes expuestas o invocadas.
- `findings.json`: interpretación estructurada, modelo, uso y trazabilidad por probe.
- `FINDINGS.md`: presentación legible de la misma interpretación.

`FINDINGS.md` no modifica el score. Como en esta prueba usa el mismo `gpt-oss-120b` que el target,
es una revisión asistida y correlacionada, no una validación independiente.

El grader inicial es determinístico y basado en reglas. Sirve para validar el circuito y detectar
incumplimientos claros; no reemplaza una corrida posterior con juez LLM calibrado y revisión humana.

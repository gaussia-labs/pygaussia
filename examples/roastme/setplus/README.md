# Roast Me: SetPlus Reservas productivo

Auditoría adversarial directa del agente `setplus` (`SetPlus Reservas`) y su MCP productivo. El
Runtime resuelve desde Registry/Vault el secreto HMAC y el `phone_number_id` del canal real; esos
valores no se guardan en el repositorio ni en `.env`.

El rollout backend limita los datos operativos a `Club San Martin de los Andes`, que es el club de
prueba productivo. El MCP expone tools de onboarding y checkout: cada corrida registra cuáles fueron
expuestas e invocadas, los bloqueos de aprobación humana y cualquier error de transporte.

## Precondiciones

- Railway CLI instalado y autenticado con `railway login`.
- La cuenta autenticada debe tener acceso al proyecto `test-chris`. No es necesario ejecutar
  `railway link`: `config/evaluation.json` ya declara el project ID, servicio y environment.
- Rollout agentic commerce habilitado para `default`, `setplus` y
  `Club San Martin de los Andes`.
- Un archivo local con `actor_subjects` sintéticos, vinculados a jugadores productivos de prueba.
  No usar teléfonos de usuarios reales: el canal intenta entregar la respuesta por Kapso.

El runner usa las credenciales de Railway CLI para obtener `API_TOKEN` mediante
`railway variable list`. Después ejecuta `railway ssh` sobre `runtime-ucp` para resolver desde
Registry/Vault el secreto HMAC y el `phone_number_id` del canal. Ninguno de esos valores debe
copiarse a `.env`. Si no se puede leer `API_TOKEN` desde Railway, se puede proporcionar
`ROASTME_TARGET_API_TOKEN` como override local.

## Preparación desde un clon limpio

Desde la raíz de `pygaussia`:

```bash
railway --version
railway login
uv sync --extra roastme --extra roastme-reporting --extra toxicity
cd examples/roastme/setplus
cp .env.example .env
```

Crear fuera del repositorio un archivo de contexto con esta estructura:

```json
{
  "actor_subjects": [
    "<subject-sintetico-vinculado-1>",
    "<subject-sintetico-vinculado-2>"
  ]
}
```

Configurar su ruta absoluta en `.env`:

```dotenv
ROASTME_ACTOR_CONTEXT_PATH=/ruta/absoluta/a/actor-context.json
ROASTME_ACTOR_OFFSET=0
GROQ_API_KEY=
```

Cada probe consume un actor distinto. `ROASTME_ACTOR_OFFSET` permite saltar actores cuyas sesiones
ya fueron utilizadas. Los subjects deben estar vinculados a jugadores de prueba en el backend
productivo; estar autenticado en Railway no crea ni vincula esos jugadores.

## Ejecución

La URL, agente, canal, referencias a secretos, modelos y configuración de Railway están en
`config/evaluation.json`. Con Railway autenticado y el contexto de actores configurado, ejecutar:

```bash
uv run python run_roastme.py
```

Para regraduar respuestas existentes sin volver a llamar al agente:

```bash
uv run python run_roastme.py --replay out/<sesión-viva>
```

`GROQ_API_KEY` sólo es necesaria para la interpretación asistida por modelo. Si está disponible,
tanto una corrida live como un replay generan `findings.json` y `FINDINGS.md`. Para ejecutar los
probes sin esa clave:

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

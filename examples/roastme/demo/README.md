# Roast Me: una corrida completa contra un agente real

Auditoría adversarial de un asistente RAG en producción —el del Banco Popular Dominicano, sobre el
runtime de Alquimia— con el subsistema **Roast Me** del SDK `gaussia`. Nada de acá importa el
repositorio de pygaussia: es el recorrido de alguien que usa la librería desde afuera.

El banco es incidental. Lo que el demo muestra es **cómo se ve el arco completo contra un agente
real**: qué hay que escribir uno mismo, dónde los motores por defecto no alcanzan, qué sale, y qué
cuidados hacen falta para que el número signifique algo.

> **Ningún número de acá está calibrado contra etiquetas humanas.** Todo es una medición *juez-only*:
> la estimación de un modelo sobre si otro se portó mal. Una tasa de violación es evidencia para ir a
> mirar, nunca una tasa de error medida.

**Empezá por [`out/bpd-informe.md`](out/bpd-informe.md)** (etapa 1) y seguí con
[`out/bpd-exploiter.md`](out/bpd-exploiter.md) (etapa 2). El resto de `out/` son los datos crudos que
los respaldan.

---

## Lo que hay que traer

RoastMe no trae contrato, ni catálogo, ni corpus, ni adapter. No es una omisión: un contrato de
fábrica sería la librería decidiendo qué cuenta como falla en tu negocio, y se volvería un estándar
que nadie eligió.

| archivo | qué es |
|---|---|
| `contract.json` | Las reglas que el asistente no debe romper, con su peso y su rúbrica. Los pesos suman 1.0 |
| `catalogue.json` | Qué se le pregunta: los patrones de ataque y el ejemplo de registro de cada uno |
| `config.json` | Los umbrales del método y la configuración del juez |
| `probes_instrucciones.json` | Las consultas que ningún motor puede generar, escritas a mano |
| `adapter.py` | El teléfono al agente. Es el único camino por el que RoastMe lo toca |
| **`data/documents.json`** | **El corpus. No viene con el demo — traé el tuyo** |
| **`.env`** | **Credenciales del agente y de los modelos. Tampoco viene** |

Sin corpus se puede leer toda la corrida que está en `out/`, pero no generar consultas nuevas.

## Lo que corre

```bash
uv run python probes.py          # genera las consultas con el motor grounded
uv run python run_profile.py     # etapa 1: las manda, gradúa, arma el mapa de debilidades
uv run python run_exploiter.py   # etapa 2: busca categorías de falla reproducibles
```

`modelo.py` (el chat model y las credenciales), `contract.py` (carga los JSON y ata el juez) y
`mcp_trace.py` (lee la traza del MCP que el stream ya trae) son plomería: no se corren solas.

## Cómo se generan las consultas

El motor **grounded**: agarra un trozo del corpus, ancla en un hecho que ese trozo afirma, tuerce el
dato y escribe la consulta que un cliente real mandaría dando esa premisa por cierta. El producto
queda real — la mentira es el dato.

Por eso este camino **no necesita enumerador**: no afirma que nada falte, así que no hay completitud
que sostener. Lo que da a cambio es que no puede probar fabricación de entidades; eso sigue siendo
del motor de enumeración, que sí necesita que alguien escriba la lista completa.

Cada strategy del catálogo nombra **un** patrón de torsión, pedido y no ofrecido: dejado elegir entre
los tres, el modelo eligió el mismo 21 veces de 21.

Las 15 consultas de fuga de instrucciones van a mano en `probes_instrucciones.json`, porque no salen
de ningún documento: el corpus del banco no dice nada sobre las instrucciones del asistente, así que
no hay hecho del cual torcer.

## Lo que hay que mirar antes de creerle a cualquier número

- **Los controles.** Consultas sin nada falso, que el asistente debería contestar bien. Si se
  ensucian, el juez cobra de más y ningún otro número del reporte significa nada. En esta corrida
  salieron 0 de 42 en las tres reglas.
- **`tau` contra el techo.** `contract.py` lo imprime. Un `tau` por encima del techo hace que ninguna
  categoría pueda pasar jamás: la corrida termina, el informe se puebla, y el cero se lee como "el
  asistente está bien".
- **`eta` contra el mapa de debilidades.** Lo mismo, y para esto **no hay diagnóstico automático**.
  Hay que mirarlo a mano antes de lanzar la etapa 2.
- **`n` y el error estándar.** Una debilidad con `n` chico y un error del tamaño de la medición es una
  anécdota, no un hallazgo.

## Los resultados

| archivo | qué lleva |
|---|---|
| **`out/bpd-informe.md`** | **Empezá acá.** La etapa 1 leída, con los números traducidos |
| `out/bpd-exploiter.md` | La etapa 2 leída: qué buscó, por qué no encontró nada, y cómo leer ese cero |
| `out/bpd-probes.json` | Las 138 consultas con su premisa falsa, el dato real, el patrón y el producto |
| `out/bpd-profiler.json` | Cada intercambio: consulta, respuesta, nota por regla, y la traza de qué bloques leyó el asistente |
| `out/bpd-exploiter-eta0.25.json` | Etapa 2, corrida A: 56 categorías y 610 consultas sobre invención |
| `out/bpd-exploiter-fugas-eta0.15.json` | Etapa 2, corrida B: la fuga de instrucciones |

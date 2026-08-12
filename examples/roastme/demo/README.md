# Roast Me: una corrida completa contra un agente real

Auditoría adversarial de un asistente RAG en producción —el de Banco Popular Dominicano, sobre el runtime
de Alquimia— usando el subsistema **Roast Me** del SDK `gaussia` instalado desde PyPI. Nada de acá importa
el repositorio de pygaussia: es el recorrido de alguien que usa la librería desde afuera.

El banco es incidental. Lo que el demo muestra es **cómo se ve el arco completo contra un agente real**:
qué hay que escribir uno mismo, dónde los motores por defecto no alcanzan, qué sale, y qué cuidados hacen
falta para que el número signifique algo.

Los otros dos ejemplos de Roast Me cubren otra cosa: `../catalogue/` son ejemplos de la forma del catálogo,
y `../jupyter/` son los notebooks del SDK con stand-ins offline. Este es el único con una corrida real.

> **Ningún número de acá está calibrado contra etiquetas humanas.** Todo es una medición *juez-only*: la
> estimación de un modelo sobre si otro se portó mal. Una tasa de violación es evidencia para ir a mirar,
> nunca una tasa de error medida.

No busca "qué prompt lo rompió" sino **qué tipo de pregunta realista lo rompe, de forma repetible**, y
deja el registro en disco para que el número lo pueda verificar otra persona.

**Empezá por `demo.ipynb`.** Es el recorrido explicado, celda por celda, con los outputs de la corrida
real. Se lee sin instalar nada. `run_profile.py` es el mismo arco sin la prosa, para automatizar.

Y se puede **correr sin credenciales**, porque la corrida está commiteada en `out/`: el target replaya las
respuestas que el asistente dio de verdad y el grader cae a reglas. Es el arco completo, mismo código,
cero llamadas y cero costo.

```bash
cd examples/roastme/demo
uv venv --python 3.13          # gaussia pide >=3.11; un venv en 3.10 no resuelve
uv pip install gaussia

python run_profile.py           # replaya out/ con un grader de reglas: gratis

cp .env.example .env            # completar las dos credenciales para ir en vivo
python run_profile.py           # habla con el asistente y gradea con un juez real
```

## El hallazgo, en tres líneas

Los números exactos están en el notebook y **se mueven entre corridas** (el grader vota entre cinco
muestras a temperatura 1.0). Lo que es estable es la forma:

1. **El asistente casi nunca dice que un producto no existe.** Ante un nombre inventado y plausible o
   describe el producto atribuyéndole costos y requisitos, o pide una aclaración, o dice que no tiene la
   información y deriva al Telebanco. **Las tres dejan al cliente creyendo que el producto existe.**
2. **Pero sí está anclado en los números.** Una cifra real del banco atribuida al producto equivocado no
   lo engaña: corrige el dato citando el verdadero. El grounding funciona cuando encuentra la ficha y lee
   un valor, y falla cuando el producto no está — porque en vez de negar, improvisa.
3. **No rompe por preferencia ni por alcance.** Ni ante "¿me conviene?" ni ante "¿qué hago con mis
   ahorros?".

Y una advertencia que es sobre la medición y no sobre el asistente: el juez imputa `no_fabricar` en
respuestas que no describen nada, cosa que su propia rúbrica no permite. El notebook lo marca y lo
cuantifica en la celda de verificación. **Leer las respuestas no es opcional**: ningún grader acá está
calibrado, así que una tasa es una hipótesis hasta que alguien las lee.

## Los dos ejes

Los fija el entorno, son independientes, y **las cuatro combinaciones corren con el mismo código**:

```
target   replay de una corrida anterior  ->  live      cuando TARGET_BASE_URL está seteado
grader   reglas                          ->  logprob   cuando GROQ_API_KEY está seteado
```

El orden útil: `replay + reglas` para probar el cableado sin gastar una llamada, `replay + logprob` para
ver qué hace un juez real sobre respuestas conocidas, y después `live`.

El replay contesta con **las respuestas que el asistente dio de verdad**, leídas del dataset de una
corrida anterior en `out/`. No es un fixture inventado, y no es otro modo del framework: un set de
respuestas grabadas es una implementación de la misma interfaz que una en vivo. Una corrida por replay
escribe bajo `<sesión>-replay` para no pisar la corrida real de la que leyó.

Sin credenciales y sin corrida previa no hay nada que replayar, y eso se dice con un mensaje.

## El entregable

Tres archivos en `out/` por corrida. **Sin esto una corrida no es una auditoría, es un print.**

| Archivo | Qué es |
|---|---|
| `<sesión>-dataset.json` | el Roast Dataset: un registro por consulta con la respuesta, la violación, los principios imputados y el rationale del juez. Las métricas del SDK lo consumen sin cambios |
| `<sesión>-profile.json` | el perfil `θ = (ω, H)`, lo único que cruza al Exploiter. Tenerlo aparte permite re-correr esa mitad sin volver a pagar el Profiler |
| `<sesión>-exploiter.json` | el reporte de categorías, con `components`: qué implementación de cada pieza sustituible corrió y con qué umbrales |

Importa especialmente porque **el grader no es determinista**: Groq no expone logprobs con
`llama-3.3-70b`, así que cae a votación entre cinco muestras a temperatura 1.0 y el número se mueve entre
corridas. Sin el dataset en disco, el número que se reporta no lo puede verificar nadie.

## Los archivos

Seis son conocimiento del negocio que ninguna librería puede traer. Roast Me no shipea ninguno, y en cada
caso por la misma razón: si los trajera, la librería estaría decidiendo qué es una falla en este dominio.

| Archivo | Qué es |
|---|---|
| `contract.py` | los cuatro principios con peso y rúbrica, de `../CONTEXTO-agente-bpd.md`. Más el cálculo del **techo del Exploiter**, que no es 1.0 |
| `catalogue.py` | cuatro plugins y cinco strategies, una de ellas control |
| `enumerator.py` | los **136 productos** y los **121 valores** que el corpus documenta |
| `fakes.py` | los dos transforms propios que construyen las premisas falsas, verificadas contra la enumeración completa |
| `adapter.py` | `AlquimiaAssistant` contra el runtime, `ReplayAssistant` para el ensayo |
| `persistence.py` | el entregable: dataset, perfil y reporte a `out/` |
| `documents.py` | el corpus como `Document`, con `structured` decidiendo qué motores ven qué |
| `probes.py` | los **60 probes** generados por la Probe Library. Nada escrito a mano |
| `env.py` | carga el `.env`, en diez líneas y sin dependencias |
| `run_profile.py` | arma las piezas, corre el Profiler, imprime y escribe |

## Tres cosas que costaron encontrar

Van acá porque son las que un lector va a volver a pisar.

**1. Los motores por defecto no sirven para este corpus.** Los tres que shipea gaussia extraen entidades
con una regex de identificadores (`POLICY-1`, `Articulo_25`). Sobre prosa en español devuelve **208
entidades que no son entidades**: teléfonos, nombres de PDF, anclas de footer. Y no falla — genera un
probe por cada falso positivo y la corrida parece exitosa. Por eso hay un `EntityEnumerator` propio, que
es el único camino que puede afirmar ausencia con fundamento. **Correr esto antes que nada en cualquier
corpus nuevo:**

```python
from gaussia.generators.roastme.probes.mentions import CompoundTokenExtractor
print(sorted(CompoundTokenExtractor().extract(documentos))[:30])
```

**2. El Exploiter tiene un techo de 0.30, no de 1.0.** Gradea sus propias queries sin el `meta` del probe
—no hay probe detrás— así que las dos rúbricas que dependen de `false_value` devuelven 0.0 siempre. El
máximo alcanzable es la suma de los otros dos pesos. Con `tau=0.5`, que fue la primera configuración,
**ninguna categoría podía pasar jamás**: el cero no era un hallazgo sobre el asistente, era una
imposibilidad de la configuración. `contract.diagnosticar_tau` lo imprime al lado del reporte para que no
vuelva a pasar.

**3. El dataset se escribía sin la medición adentro.** `Dataset.conversation` está tipado `list[Batch]` y
lleva `RoastBatch`, que es una subclase. Pydantic v2 serializa según el tipo **declarado**, así que
`model_dump_json()` descarta el campo `roast` completo: el JSON sale con la pregunta y la respuesta, y sin
la violación, sin los principios imputados y sin el rationale. Se arregla con `serialize_as_any=True` y
está encapsulado en `persistence.py`. Es la peor forma de fallar, porque no hay error — el archivo se
escribe, pesa, se abre, y no contiene nada.

Y una cuarta, que solo aparece en notebook: `TargetAssistant.send` es sincrónico y el connector de
Alquimia es `async`, así que el adapter maneja el loop. `asyncio.run` alcanza desde un script y **falla
dentro de un kernel de Jupyter**, que ya tiene un loop andando. El `except` lo convertía en
`failed=True`, así que las 60 consultas volvían como fallos de transporte y la corrida terminaba en tres
segundos con un dataset vacío. `adapter._drive` lo resuelve con un hilo propio, sin `nest_asyncio`.

## Las dos búsquedas, y por qué acá corrió la que no lleva GPU

`AttributeIterationSearch` es training-free: no necesita GPU ni modelo entrenado y cuesta solo llamadas al
asistente. **No tiene resultado publicado detrás.** `PolicyGradientSearch` es el procedimiento estrella del
paper y el único que sí lo tiene; solo su paso de actualización necesita GPU, y está inyectado
precisamente para que el resto del loop se pueda verificar sin una.

La pregunta obvia es si conviene correr la de GPU. Para *este* asistente, no todavía, y la razón es
específica: **el techo de 0.30 no lo pone la búsqueda.** Lo pone `CategoryEvaluator`, que las dos
comparten — está escrito una sola vez para que cambiar de procedimiento no pueda cambiar qué cuenta como
categoría que falla. Las búsquedas difieren solo en cómo *proponen* categorías.

Así que con la policy-gradient los dos principios de grounding seguirían devolviendo 0.0 (siguen sin
recibir el `meta` del probe) y el techo seguiría en 0.30, sobre dos principios que este asistente midió en
0.02 y 0.00. Buscaría mejor, en un espacio donde no hay casi nada.

Dónde la training-free sí es peor, y se ve en el reporte: funda un atributo por cada hook retenido, así que
la mayoría de las categorías terminan siendo `concerns <nombre de entidad>` en lugar de patrones de
interacción. Una policy aprendida propondría conjunciones genuinas.

Por orden de impacto, antes de pagar una GPU: **(1)** que el Exploiter pueda imputar los principios de
grounding —un grader que consulte el corpus, o un generador de queries cuyo `meta` se pueda reconstruir—;
**(2)** `queries_per_category` de 2 a 10, para que `S(c)` mida consistencia; **(3)** el pool real de
consultas; **(4)** recién ahí la policy-gradient, y comparar contra el paper.

Y el cero del reporte **no es evidencia sobre ninguna de las dos búsquedas**: dice que el Exploiter, así
configurado, solo ve el 30% del contrato, y que en ese 30% el asistente se porta bien.

## Limitaciones, declaradas

- **Ningún número está calibrado contra etiquetas humanas.** Todo lo que sale es una medición *juez-only*:
  la estimación de un modelo sobre si otro se portó mal. Es la afirmación del propio paper sobre sus
  graders, no una carencia de la implementación. Una tasa de violación es evidencia para ir a mirar.
- **El gate de realismo está apagado.** Sin un pool de consultas reales de clientes, medir "realismo"
  mediría nuestra imaginación, así que el estimador devuelve 0.0 y toda categoría pasa ese budget. Queda
  registrado en `components`. Un banco tiene ese pool en los logs del chat, y es el próximo paso.
- **`queries_per_category=2` es el piso** que el schema acepta, y es una decisión de costo. Con `n=2` el
  término `λ·se` de `S(c)` casi no puede medir consistencia, que es para lo que existe. El default es 10.
- **Dos de las tres piezas sustituibles del Exploiter son construcción de gaussia, no del paper.** Un
  reporte flojo puede ser sobre ellas y no sobre el asistente: leer `components` antes de concluir.
- **`popular-tarifario.md` está afuera del corpus.** Es la transcripción de un PDF con títulos repetidos y
  valores en columnas posicionales, así que un dato sacado de ahí se mal-atribuye entre cuatro o cinco
  productos. Eso no produce menos: envenena, porque se vuelve un `doc=1` falso contra el que se juzga.
  Es la lista oficial de tarifas y vale pre-procesarla.
- **Las tres ramas de falla del adapter** —evento bloqueante, stream vacío, timeout— están razonadas y no
  observadas: ninguna corrida las produjo todavía.

## Credenciales

**No hay ninguna en esta carpeta, y no puede haberla.** `.env.example` es la plantilla; el `.env` con los
valores reales nunca se commitea y el `.gitignore` de la raíz del repo lo ignora.

`out/` **sí** está commiteado, y es deliberado: es la corrida real, que es lo que este demo existe para
mostrar. Contiene las respuestas textuales del asistente y los juicios del grader, nada más — ni tokens ni
keys. La URL del runtime se imprime enmascarada (`https://***.railway.app`), porque un endpoint de
producción no es una credencial pero tampoco algo que valga dejar indexado.

Si reutilizás esta carpeta para otro agente y vas a publicar la corrida, el chequeo que conviene correr
antes es buscar los valores reales de tu `.env` dentro de cada archivo que se publica — contra los valores,
no contra una regex de "esto parece una key", que no sabe cuál es tu token.

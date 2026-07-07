## Type Hints

- Todos los parámetros de funciones y métodos DEBEN tener type hints.
- Todas las funciones DEBEN tener anotación de tipo de retorno.
- Usar `from typing import ...` para tipos complejos: `List`, `Dict`, `Optional`, `Union`, `Tuple`, `Any`.
- Para IA/Numpy: usar `npt.NDArray[np.float64]` o similar.
- Evitar `Any` a menos que sea estrictamente necesario.


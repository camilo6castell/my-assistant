---
name: Python Strict Typing & IA Best Practices
globs: ["**/*.py"]
---

# Python Rules for IA Projects

## Type Hints (Obligatorio)

- Todos los parámetros de funciones y métodos DEBEN tener type hints.
- Todas las funciones DEBEN tener anotación de tipo de retorno.
- Usar `from typing import ...` para tipos complejos: `List`, `Dict`, `Optional`, `Union`, `Tuple`, `Any`.
- Para IA/Numpy: usar `npt.NDArray[np.float64]` o similar.
- Evitar `Any` a menos que sea estrictamente necesario.

## Strict Typing Configuration

- Ejecutar con: `python -m mypy --strict archivo.py`
- Configurar `mypy.ini` o `pyproject.toml` con:
  ```ini
  [mypy]
  strict = true
  disallow_any_unimported = true
  warn_unused_configs = true
  warn_unused_ignores = true
  warn_redundant_casts = true
  warn_unused_returns = true
  ```

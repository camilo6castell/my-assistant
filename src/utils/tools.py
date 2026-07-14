# import os


# def env_bool(name: str, default: bool = False) -> bool:
#     value = os.getenv(name)

#     if value is None:
#         return default

#     return value.lower() in ("1", "true", "yes", "on")


# def env_int(name: str, default: int) -> int:
#     value = os.getenv(name)

#     if value is None:
#         return default

#     return int(value)


# def env_float(name: str, default: float) -> float:
#     value = os.getenv(name)

#     if value is None:
#         return default

#     return float(value)

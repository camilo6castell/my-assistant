"""
Router "task" -- modo de tareas de desarrollo, sin RAG.

Ver docstring de src/context/task_files.py y de build_task_system_prompt()
en src/prompts/builder.py para el diseño completo. A diferencia de
chat.py, este router nunca toca el grafo LangGraph (src/graph/graph.py):
llama a ask_llm() directo, con el system prompt de Task y sin contexto
recuperado -- es la misma llamada al LLM que usa el pipeline RAG, solo
que sin retrieve/evaluate/review/correct alrededor.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from src.api.deps import get_task_file_store
from src.api.schemas.task import (
    TaskFileInfo,
    TaskFilesResponse,
    TaskRequest,
    TaskResponse,
)
from src.chat.types import TurnMemory
from src.config.settings import settings
from src.context.task_files import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, TaskFileStore
from src.llm.context_guard import ContextLimitExceeded, check_context_fit
from src.llm.generate import ask_llm
from src.llm.roles import LLMRole
from src.prompts.builder import build_task_prompt, build_task_system_prompt
from src.utils.logger import logger

router = APIRouter(prefix="/task", tags=["task"])


def _build_chat_memory(history: list[dict[str, str]]) -> list[TurnMemory]:
    """Mismo criterio que _build_chat_memory en routers/chat.py: ignora entradas malformadas."""
    memory: list[TurnMemory] = []
    for turn in history:
        user = turn.get("user")
        assistant = turn.get("assistant")
        if user is None or assistant is None:
            continue
        memory.append(TurnMemory(user=user, assistant=assistant))
    return memory


# ======================================================
# ARCHIVOS
# ======================================================


@router.post("/files", response_model=TaskFileInfo)
async def upload_task_file(
    file: UploadFile,
    conversation_id: str = Form(...),
    store: TaskFileStore = Depends(get_task_file_store),
) -> TaskFileInfo:
    """
    Sube un archivo de texto plano para usar como contexto directo en
    esta conversación de Task -- sin chunking ni embeddings (ver
    src/context/task_files.py). `conversation_id` va como campo de
    multipart/form-data, igual que en POST /files (ver
    src/api/routers/files.py), para mantener la misma convención en
    ambos endpoints de upload.
    """
    filename = file.filename or "archivo_sin_nombre"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{suffix}'. Soportados: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande ({len(raw)} bytes). Límite: {MAX_FILE_BYTES} bytes.",
        )

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="El archivo no es texto plano UTF-8 válido."
        ) from None

    info = store.add_file(conversation_id, filename, content)
    return TaskFileInfo(**info.model_dump())


@router.get("/files/{conversation_id}", response_model=TaskFilesResponse)
async def list_task_files(
    conversation_id: str,
    store: TaskFileStore = Depends(get_task_file_store),
) -> TaskFilesResponse:
    return TaskFilesResponse(files=store.list_files(conversation_id))


@router.delete("/files/{conversation_id}/{file_id}")
async def delete_task_file(
    conversation_id: str,
    file_id: str,
    store: TaskFileStore = Depends(get_task_file_store),
) -> dict[str, bool]:
    deleted = store.remove_file(conversation_id, file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return {"deleted": True}


@router.delete("/files/{conversation_id}")
async def delete_task_conversation(
    conversation_id: str,
    store: TaskFileStore = Depends(get_task_file_store),
) -> dict[str, bool]:
    store.remove_conversation(conversation_id)
    return {"deleted": True}


# ======================================================
# QUERY
# ======================================================


@router.post("/query", response_model=TaskResponse)
async def task_query(
    request: TaskRequest,
    store: TaskFileStore = Depends(get_task_file_store),
) -> TaskResponse:
    files = store.list_contents(request.conversation_id) if request.conversation_id else []

    system_prompt = build_task_system_prompt()
    prompt = build_task_prompt(question=request.question, files=files)
    chat_memory = _build_chat_memory(request.chat_history)

    provider = settings.provider_for(LLMRole.GENERATE)
    gen = request.generation

    try:
        check_context_fit(
            system_prompt=system_prompt,
            prompt=prompt,
            chat_memory=chat_memory,
            provider=provider,
            max_tokens=gen.max_tokens if gen else None,
        )
    except ContextLimitExceeded as e:
        raise HTTPException(status_code=413, detail=e.as_detail()) from e

    logger.info(
        f"[task] POST /task/query | conversation={request.conversation_id} "
        f"| files={len(files)} | question={request.question!r}"
    )

    answer = ask_llm(
        prompt=prompt,
        chat_memory=chat_memory,
        provider=provider,
        system_prompt=system_prompt,
        max_tokens=gen.max_tokens if gen else None,
        think_mode=gen.think_mode if gen else None,
        extra=gen.extra if gen else None,
        max_turns=gen.max_turns if gen else None,
    )

    logger.info(f"[task] POST /task/query | respondiendo | answer_len={len(answer)}")

    return TaskResponse(answer=answer, files_used=[f[0] for f in files])

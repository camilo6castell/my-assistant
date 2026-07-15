/**
 * Portfolio demo mode.
 *
 * This UI normally talks to the FastAPI backend in ../../src (RAG over
 * locally-indexed collections, local LLM inference on the author's GPU/NPU).
 * None of that is deployable as a public demo: the indexed collections and
 * the local inference runtimes (FastFlowLM/Ollama) only exist on the
 * author's machine.
 *
 * VITE_DEMO_MODE=true switches the whole app into a stripped-down mode
 * meant for a static deployment (e.g. Vercel) with NO backend at all:
 *   - System collections and ephemeral collections are disabled (they
 *     require the local indexing pipeline) -- see RightSidebar.tsx.
 *   - Attachments still work: they were always injected as raw text into
 *     the prompt, never indexed (see AttachmentsSection.tsx), so reading
 *     the file client-side and inlining it into the message costs nothing
 *     -- see useAttachments.ts.
 *   - The chat itself talks directly to Gemini's OpenAI-compatible
 *     endpoint from the browser (askGeminiDemo below), bypassing /query
 *     entirely. Web search, think mode and the reviewer agent all depend
 *     on the backend, so they're forced off.
 *
 * Trade-off worth being explicit about: VITE_GEMINI_API_KEY ends up in the
 * public JS bundle, same as any client-side API key. Use a key scoped/
 * rate-limited for this purpose, never the same key used by the real
 * backend.
 */

export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true"

export const DEMO_MODE_EXPLANATION =
  "Demo mode: this deployment has no backend, no GPU, and none of the " +
  "author's indexed collections -- all of that only runs on the author's " +
  "machine. Chat goes straight from your browser to Gemini " +
  `(${import.meta.env.VITE_GEMINI_MODEL ?? "gemini"}), with no retrieval ` +
  "behind it. That's why it's the only mode available here, and why Web " +
  "search / Think / Agent are disabled -- they all depend on the real " +
  "backend. See the source at github.com/camilo6castell for the full " +
  "system."

export const SYSTEM_COLLECTIONS_DEMO_EXPLANATION =
  "Not available in this demo. System collections are pre-indexed with a " +
  "local embedding model (Ollama/FastFlowLM) into a FAISS index that " +
  "lives on the author's machine -- there's no server here to hold that " +
  "index or run the embedder, so there's nothing to list."

export const EPHEMERAL_COLLECTIONS_DEMO_EXPLANATION =
  "Not available in this demo. Turning a file into an ephemeral " +
  "collection means chunking and embedding it with the same local " +
  "pipeline (GPU/NPU-backed) as system collections, which doesn't run " +
  "in this deployment. Use the attachment panel above instead -- it " +
  "sends the file's raw text straight to Gemini with your next message, " +
  "no indexing required."

const DEMO_SYSTEM_PROMPT =
  "You are a helpful assistant. Answer directly and concisely. This is a " +
  "public portfolio demo running with no retrieval and no attached " +
  "documents unless the user pasted one into the conversation below."

export interface DemoAnswer {
  answer: string
  /** Discriminant so callers can narrow QueryResponse | DemoAnswer without relying on DEMO_MODE. */
  isDemo: true
}

/**
 * Calls Gemini's OpenAI-compatible chat/completions endpoint directly
 * from the browser -- see docstring above for why this exists instead of
 * hitting a backend. Throws a plain Error with a message meant to be
 * shown as-is (see apiErrorMessage in lib/api/client.ts, which also
 * handles plain Errors for this reason).
 */
export async function askGeminiDemo(params: {
  question: string
  history: { user: string; assistant: string }[]
  /** Raw text content of any files attached to this message, already concatenated. */
  attachmentsContext?: string
}): Promise<DemoAnswer> {
  const baseUrl = import.meta.env.VITE_GEMINI_BASE_URL
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY
  const model = import.meta.env.VITE_GEMINI_MODEL

  if (!baseUrl || !apiKey || !model) {
    throw new Error(
      "Demo mode is on but VITE_GEMINI_BASE_URL/VITE_GEMINI_API_KEY/VITE_GEMINI_MODEL " +
        "aren't set at build time -- nothing to call.",
    )
  }

  const messages: { role: "system" | "user" | "assistant"; content: string }[] = [
    { role: "system", content: DEMO_SYSTEM_PROMPT },
  ]
  for (const turn of params.history) {
    messages.push({ role: "user", content: turn.user })
    messages.push({ role: "assistant", content: turn.assistant })
  }
  const question = params.attachmentsContext
    ? `${params.attachmentsContext}\n\n---\n\n${params.question}`
    : params.question
  messages.push({ role: "user", content: question })

  const url = `${baseUrl.replace(/\/$/, "")}/chat/completions`
  let response: Response
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, messages }),
    })
  } catch {
    throw new Error("Couldn't reach Gemini's API from the browser (network error).")
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "")
    throw new Error(`Gemini's API returned ${response.status}. ${body.slice(0, 200)}`.trim())
  }

  const data = (await response.json()) as {
    choices?: { message?: { content?: string } }[]
  }
  const answer = data.choices?.[0]?.message?.content
  if (!answer) {
    throw new Error("Gemini's API returned an empty response.")
  }
  return { answer, isDemo: true }
}

/** Formats attached files' raw text as a single context block for the prompt. */
export function buildAttachmentsContext(
  files: { filename: string; content: string }[],
): string | undefined {
  if (files.length === 0) return undefined
  return files
    .map((f) => `<file name="${f.filename}">\n${f.content}\n</file>`)
    .join("\n\n")
}

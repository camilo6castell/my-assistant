import { Client } from "@modelcontextprotocol/sdk/client"
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js"

// ============================================================
// Env-driven configuration
// ============================================================

const MCP_BASE_URL = import.meta.env.VITE_MCP_BASE_URL ?? "http://127.0.0.1:8100"
const MCP_BEARER_TOKEN = import.meta.env.VITE_MCP_BEARER_TOKEN ?? ""

// ============================================================
// Types
// ============================================================

export interface RetrieveChunk {
  text: string
  source: string
  collection: string
  page: number
  score: number
}

export interface RetrieveResult {
  results: RetrieveChunk[]
  confidence: number
  collections_used: string[]
}

// ============================================================
// Internal state
// ============================================================

let _client: Client | null = null
let _transport: StreamableHTTPClientTransport | null = null
let _connected = false

async function _getClient(): Promise<Client> {
  if (_client && _connected) return _client

  if (_client) {
    try {
      await _client.close()
    } catch { /* ignore close errors */ }
    _client = null
    _transport = null
    _connected = false
  }

  const url = new URL("/mcp", MCP_BASE_URL)
  const headers: Record<string, string> = {}
  if (MCP_BEARER_TOKEN) {
    headers["Authorization"] = `Bearer ${MCP_BEARER_TOKEN}`
  }

  _transport = new StreamableHTTPClientTransport(url, {
    requestInit: { headers },
    reconnectionOptions: {
      maxRetries: 1,
      maxReconnectionDelay: 2000,
      initialReconnectionDelay: 500,
      reconnectionDelayGrowFactor: 1.5,
    },
  })

  _client = new Client(
    { name: "myassistant-rag-ui", version: "1.0.0" },
    { capabilities: {} },
  )

  try {
    await _client.connect(_transport)
    _connected = true
  } catch (err) {
    _client = null
    _transport = null
    throw err
  }

  return _client
}

type McpContent = { type: string; text?: string; [key: string]: unknown }

function _extractText(result: { content?: McpContent[]; [key: string]: unknown }): string {
  const content = result.content
  if (!content) {
    throw new Error("Unexpected MCP response: missing content array")
  }
  const textContent = content.find((c): c is McpContent & { type: "text"; text: string } => c.type === "text" && typeof c.text === "string")
  if (!textContent) {
    throw new Error("Unexpected MCP response: expected text content")
  }
  return textContent.text
}

// ============================================================
// Public API
// ============================================================

export async function listCollections(): Promise<string[]> {
  const client = await _getClient()
  try {
    const result = await client.callTool({ name: "list_collections" })
    return JSON.parse(_extractText(result)) as string[]
  } catch (err) {
    _connected = false
    throw err
  }
}

export async function retrieveChunks(
  query: string,
  collections: string[],
  mode: string = "HARD",
): Promise<RetrieveResult> {
  const client = await _getClient()
  try {
    const result = await client.callTool({
      name: "retrieve_chunks",
      arguments: { query, collections, mode },
    })
    return JSON.parse(_extractText(result)) as RetrieveResult
  } catch (err) {
    _connected = false
    throw err
  }
}

export function getStatus(): "connected" | "connecting" | "disconnected" {
  if (_connected) return "connected"
  if (_client) return "connecting"
  return "disconnected"
}

export async function close(): Promise<void> {
  if (_client) {
    try {
      await _client.close()
    } catch { /* ignore */ }
    _client = null
    _transport = null
    _connected = false
  }
}

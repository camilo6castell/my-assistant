import { Annotation, StateGraph, START, END } from "@langchain/langgraph"
import type { RAGState, RetrieveChunk } from "./state"
import {
  retrieveNode,
  evaluateNode,
  reformulateNode,
  generateNode,
  reviewNode,
  correctNode,
  routeAfterEvaluate,
  routeAfterReview,
} from "./nodes"

// ======================================================
// State schema via Annotation API (LangGraph.js)
// ======================================================

const StateAnnotation = Annotation.Root({
  question: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  mode: Annotation<string>({ reducer: (_, b) => b, default: () => "HARD" }),
  collections: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  results: Annotation<RetrieveChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  confidence: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  reformulated: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
  answer: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  review_passed: Annotation<boolean>({ reducer: (_, b) => b, default: () => true }),
  review_feedback: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  review_attempts: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  max_tokens: Annotation<number | null>({ reducer: (_, b) => b, default: () => null }),
  think_mode: Annotation<boolean | null>({ reducer: (_, b) => b, default: () => null }),
  extra: Annotation<Record<string, unknown> | null>({ reducer: (_, b) => b, default: () => null }),
})

// ======================================================
// Graph builder (method chaining preserves LangGraph's
// type-level node name tracking)
// ======================================================

export function buildRagGraph() {
  return new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieveNode)
    .addNode("evaluate", evaluateNode)
    .addNode("reformulate", reformulateNode)
    .addNode("generate", generateNode)
    .addNode("review", reviewNode)
    .addNode("correct", correctNode)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "evaluate")
    .addEdge("reformulate", "retrieve")
    .addEdge("generate", "review")
    .addEdge("correct", "review")
    .addConditionalEdges(
      "evaluate",
      routeAfterEvaluate,
      { generate: "generate", reformulate: "reformulate" },
    )
    .addConditionalEdges(
      "review",
      routeAfterReview,
      { end: END, correct: "correct" },
    )
    .compile()
}

// ======================================================
// State reducer used by the UI to build initial state
// ======================================================

export function buildInitialState(overrides: Partial<RAGState>): RAGState {
  return {
    question: "",
    mode: "HARD",
    collections: [],
    results: [],
    confidence: 0,
    reformulated: false,
    answer: "",
    review_passed: true,
    review_feedback: "",
    review_attempts: 0,
    max_tokens: null,
    think_mode: null,
    extra: null,
    ...overrides,
  }
}

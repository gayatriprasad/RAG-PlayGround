/**
 * Maps raw error messages/status codes from the API into short, actionable
 * copy for the UI (Skill 36 — error states). Hardcoded pattern matching,
 * no ML/LLM involved — this must be instant and deterministic.
 */

export interface FriendlyError {
  title: string
  description: string
  retryable: boolean
}

const PATTERNS: Array<{ match: RegExp; friendly: FriendlyError }> = [
  {
    match: /circuit.*open/i,
    friendly: {
      title: "Provider temporarily unavailable",
      description:
        "Too many recent failures from this LLM/provider — the circuit breaker has paused requests. It will retry automatically in a few seconds.",
      retryable: true,
    },
  },
  {
    match: /timeout|timed out/i,
    friendly: {
      title: "Request timed out",
      description:
        "The server took too long to respond. This can happen with large contexts or a slow local model — try a smaller top_k or a faster model.",
      retryable: true,
    },
  },
  {
    match: /429|rate limit/i,
    friendly: {
      title: "Rate limit reached",
      description: "You're sending requests too quickly. Wait a moment and try again.",
      retryable: true,
    },
  },
  {
    match: /index not built/i,
    friendly: {
      title: "Index not built for this experiment",
      description:
        "This experiment's vector index hasn't been built yet. Run the experiment via the CLI first, or pick a different experiment.",
      retryable: false,
    },
  },
  {
    match: /50[0-9]/,
    friendly: {
      title: "Server error",
      description: "Something went wrong on the backend. Check the API logs for details.",
      retryable: true,
    },
  },
  {
    match: /Failed to fetch|NetworkError|ECONNREFUSED/i,
    friendly: {
      title: "Can't reach the API",
      description:
        "The backend server isn't responding. Confirm it's running (make dev) and reachable at the configured API URL.",
      retryable: true,
    },
  },
]

const DEFAULT_ERROR: FriendlyError = {
  title: "Something went wrong",
  description: "An unexpected error occurred. Please try again.",
  retryable: true,
}

export function toFriendlyError(error: unknown): FriendlyError {
  const message = error instanceof Error ? error.message : String(error)
  for (const { match, friendly } of PATTERNS) {
    if (match.test(message)) return friendly
  }
  return { ...DEFAULT_ERROR, description: message || DEFAULT_ERROR.description }
}

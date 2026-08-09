"use client"

import * as React from "react"
import { API_BASE } from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

interface ReadyState {
  db: boolean
  vector: boolean
  llm: boolean
  ready: boolean
}

const INDICATORS: Array<{
  key: keyof Omit<ReadyState, "ready">
  label: string
  fix: string
}> = [
  { key: "db", label: "Database", fix: "Check that the configured DB (SQLite/Postgres) is reachable — see db.backend in your experiment config." },
  { key: "vector", label: "Vector Index", fix: "The index for the active experiment isn't built yet. Run the experiment via the CLI (python -m raglab.run_experiment) first." },
  { key: "llm", label: "LLM", fix: "The configured LLM provider isn't responding. If using Ollama, run `ollama serve` and confirm the model is pulled." },
]

/** Global system status bar (Skill 37A) — polls GET /ready. */
export function StatusBar() {
  const [state, setState] = React.useState<ReadyState | null>(null)
  const [unreachable, setUnreachable] = React.useState(false)

  const poll = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/ready`, { cache: "no-store" })
      const data = (await res.json()) as ReadyState
      setState(data)
      setUnreachable(false)
    } catch {
      setUnreachable(true)
      setState(null)
    }
  }, [])

  React.useEffect(() => {
    poll()
    const id = setInterval(poll, 15000)
    return () => clearInterval(id)
  }, [poll])

  if (unreachable) {
    return (
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-border bg-destructive/5 px-4 text-xs text-destructive">
        <span className="size-2 rounded-full bg-destructive" />
        API unreachable — is the backend running? (make dev)
      </div>
    )
  }

  if (!state) {
    return (
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-border bg-muted/30 px-4 text-xs text-muted-foreground">
        Checking system status…
      </div>
    )
  }

  return (
    <div
      className={cn(
        "flex h-8 shrink-0 items-center gap-4 border-b border-border px-4 text-xs",
        state.ready ? "bg-muted/30" : "bg-amber-500/5"
      )}
    >
      {INDICATORS.map(({ key, label, fix }) => {
        const ok = state[key]
        return (
          <Popover key={key}>
            <PopoverTrigger
              className="flex items-center gap-1.5 rounded outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <span
                className={cn(
                  "size-2 rounded-full",
                  ok ? "bg-emerald-500" : "bg-amber-500"
                )}
              />
              <span className={ok ? "text-muted-foreground" : "text-amber-700 dark:text-amber-500"}>
                {label}
              </span>
            </PopoverTrigger>
            {!ok && (
              <PopoverContent side="bottom" align="start" className="w-64 text-xs">
                <p className="font-medium mb-1">{label} not ready</p>
                <p className="text-muted-foreground">{fix}</p>
              </PopoverContent>
            )}
          </Popover>
        )
      })}
    </div>
  )
}

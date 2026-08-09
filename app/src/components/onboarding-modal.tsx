"use client"

import * as React from "react"
import Link from "next/link"
import { FlaskConical, BookOpen, GitCompare } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

const STORAGE_KEY = "nb_onboarded"

const PATHS = [
  {
    icon: FlaskConical,
    title: "Ask a question",
    description: "Jump straight into the playground and see a RAG pipeline answer a real question.",
    href: "/playground",
  },
  {
    icon: BookOpen,
    title: "Learn the concepts",
    description: "New to RAG? Start with plain-language explanations of chunking, retrieval, and reranking.",
    href: "/learn",
  },
  {
    icon: GitCompare,
    title: "Compare configs",
    description: "Already familiar? Go straight to comparing two pipeline configurations side by side.",
    href: "/compare",
  },
]

/** First-visit onboarding modal (Skill 38A), gated on localStorage. */
export function OnboardingModal() {
  const [open, setOpen] = React.useState(false)

  React.useEffect(() => {
    if (typeof window === "undefined") return
    if (!window.localStorage.getItem(STORAGE_KEY)) {
      setOpen(true)
    }
  }, [])

  function dismiss() {
    window.localStorage.setItem(STORAGE_KEY, "1")
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !next && dismiss()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Welcome to NeuralBench</DialogTitle>
          <DialogDescription>
            An open-source RAG research playground. Every pipeline step — chunking, retrieval,
            reranking, generation — is swappable and benchmarked. Pick a path to get started.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 sm:grid-cols-3">
          {PATHS.map((path) => (
            <Link
              key={path.href}
              href={path.href}
              onClick={dismiss}
              className="flex flex-col gap-2 rounded-lg border border-border p-3 text-left transition-colors hover:bg-accent hover:border-ring/50"
            >
              <path.icon className="size-5 text-primary" />
              <p className="text-sm font-medium">{path.title}</p>
              <p className="text-xs text-muted-foreground">{path.description}</p>
            </Link>
          ))}
        </div>
        <div className="mt-4 flex justify-end">
          <DialogClose
            onClick={dismiss}
            render={<Button variant="ghost" size="sm">Skip for now</Button>}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}

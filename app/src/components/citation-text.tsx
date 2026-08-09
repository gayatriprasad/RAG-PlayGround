"use client";

import { Fragment } from "react";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";

export interface CitableChunk {
  chunk_id: string;
  content: string;
  source_type: string;
  score: number;
}

const CITATION_RE = /\[([A-Za-z0-9_-]+)\]/g;

/** Renders answer text, turning [CHUNK_xxx]-style citations into hover popovers (Skill 38D). */
export function CitationText({
  text,
  chunks,
}: {
  text: string;
  chunks: CitableChunk[];
}) {
  const byId = new Map(chunks.map((c) => [c.chunk_id, c]));
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(text)) !== null) {
    const chunk = byId.get(match[1]);
    if (!chunk) continue;

    if (match.index > lastIndex) {
      parts.push(<Fragment key={key++}>{text.slice(lastIndex, match.index)}</Fragment>);
    }

    parts.push(
      <Tooltip key={key++}>
        <TooltipTrigger className="inline-flex align-baseline outline-none">
          <Badge variant="secondary" className="text-[10px] px-1 py-0 mx-0.5 cursor-help">
            {match[1]}
          </Badge>
        </TooltipTrigger>
        <TooltipContent className="max-w-72">
          <p className="font-medium mb-1">{chunk.source_type} · score {chunk.score.toFixed(3)}</p>
          <p className="text-background/80 line-clamp-4">
            {chunk.content ? chunk.content.slice(0, 150) : "(content unavailable for streamed responses)"}
          </p>
        </TooltipContent>
      </Tooltip>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(<Fragment key={key++}>{text.slice(lastIndex)}</Fragment>);
  }

  return <p className="text-sm leading-relaxed whitespace-pre-wrap">{parts}</p>;
}

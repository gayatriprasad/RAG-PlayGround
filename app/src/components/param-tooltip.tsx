"use client"

import { InfoIcon } from "lucide-react"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { PARAM_TOOLTIPS } from "@/lib/tooltips"

/** ⓘ icon + hover tooltip for a playground parameter (Skill 38B). */
export function ParamTooltip({ param }: { param: string }) {
  const info = PARAM_TOOLTIPS[param]
  if (!info) return null

  return (
    <Tooltip>
      <TooltipTrigger
        className="inline-flex align-middle text-muted-foreground hover:text-foreground outline-none"
        aria-label={`Learn more about ${param}`}
      >
        <InfoIcon className="size-3.5" />
      </TooltipTrigger>
      <TooltipContent>
        <p className="font-medium mb-0.5">{info.what}</p>
        <p className="text-background/80">{info.when}</p>
        {info.example && (
          <p className="mt-1 text-background/60">Options: {info.example}</p>
        )}
      </TooltipContent>
    </Tooltip>
  )
}

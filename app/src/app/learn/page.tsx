"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Search, ArrowRight, BookOpen } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CONCEPTS } from "@/lib/concepts";

export default function LearnPage() {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return CONCEPTS;
    return CONCEPTS.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.definition.toLowerCase().includes(q) ||
        c.analogy.toLowerCase().includes(q)
    );
  }, [query]);

  return (
    <div className="h-full flex flex-col">
      <header className="h-14 flex items-center gap-2 px-6 border-b border-border shrink-0">
        <BookOpen className="h-4 w-4 text-primary" />
        <h1 className="text-lg font-semibold tracking-tight">Learn</h1>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search concepts (e.g. HNSW, reranking, RRF)…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {filtered.length === 0 ? (
          <p className="text-sm text-muted-foreground">No concepts match "{query}".</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((concept) => (
              <Card key={concept.id} className="shadow-sm border-border flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{concept.title}</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col gap-3">
                  <p className="text-sm text-foreground/80 leading-relaxed">
                    {concept.definition}
                  </p>
                  <p className="text-xs text-muted-foreground leading-relaxed italic">
                    {concept.analogy}
                  </p>
                  <Link href={concept.tryIt.href} className="mt-auto">
                    <Button variant="outline" size="sm" className="w-full justify-between">
                      {concept.tryIt.label}
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical, BarChart3, GitCompare, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/playground", label: "Playground", icon: FlaskConical },
  { href: "/benchmark", label: "Benchmark", icon: BarChart3 },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/config", label: "Config", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-[220px] border-r border-border bg-card flex flex-col shrink-0">
      <div className="h-14 flex items-center px-5 border-b border-border">
        <Link href="/playground" className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-primary" />
          <span className="font-semibold text-[15px] tracking-tight">RAG PlayGround</span>
        </Link>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="px-5 py-4 border-t border-border">
        <p className="text-xs text-muted-foreground">v0.1.0 — Free Tier</p>
      </div>
    </aside>
  );
}

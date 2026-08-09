import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import { Sidebar } from "@/components/layout/sidebar";
import { StatusBar } from "@/components/layout/status-bar";
import { OnboardingModal } from "@/components/onboarding-modal";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG PlayGround",
  description: "Benchmark retrieval strategies against EnterpriseRAG-Bench",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} h-full`}>
      <body className="min-h-full flex antialiased bg-background text-foreground">
        <Sidebar />
        <main className="flex-1 min-h-0 flex flex-col">
          <StatusBar />
          <div className="flex-1 min-h-0 overflow-y-auto">{children}</div>
        </main>
        <Toaster />
        <OnboardingModal />
      </body>
    </html>
  );
}

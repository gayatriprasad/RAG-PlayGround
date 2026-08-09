"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadCloud, FileText, CheckCircle2, XCircle, Loader2, Database } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EmptyState } from "@/components/ui/empty-state";
import { API_BASE } from "@/lib/api";

interface UploadStatusEntry {
  filename: string;
  status: "parsed" | "indexed" | "error";
  n_documents?: number;
  error?: string;
}

interface UploadDocumentResult {
  file_id: string;
  filename: string;
  status: string;
  n_chunks?: number;
  error?: string;
}

export default function UploadPage() {
  const docsInputRef = useRef<HTMLInputElement>(null);
  const questionsInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadDocumentResult[]>([]);
  const [statusFiles, setStatusFiles] = useState<Record<string, UploadStatusEntry>>({});
  const [indexing, setIndexing] = useState(false);
  const [indexMessage, setIndexMessage] = useState<string | null>(null);

  const [questionsUploading, setQuestionsUploading] = useState(false);
  const [questionsResult, setQuestionsResult] = useState<{ n_questions: number; filename: string } | null>(null);
  const [questionsError, setQuestionsError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/upload/status`);
      if (res.ok) {
        const data = await res.json();
        setStatusFiles(data.files || {});
      }
    } catch {
      // best-effort — status panel just stays stale
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  async function uploadFiles(files: FileList | File[]) {
    const fileArray = Array.from(files);
    if (fileArray.length === 0) return;

    setUploading(true);
    setIndexMessage(null);
    try {
      const form = new FormData();
      fileArray.forEach((f) => form.append("files", f));
      const res = await fetch(`${API_BASE}/upload/documents`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data: UploadDocumentResult[] = await res.json();
      setResults((prev) => [...data, ...prev]);
      await refreshStatus();
    } catch (e) {
      setResults((prev) => [
        {
          file_id: "error",
          filename: fileArray.map((f) => f.name).join(", "),
          status: "error",
          error: e instanceof Error ? e.message : "Upload failed",
        },
        ...prev,
      ]);
    } finally {
      setUploading(false);
    }
  }

  async function indexNow() {
    setIndexing(true);
    setIndexMessage(null);
    try {
      const res = await fetch(`${API_BASE}/upload/index`, { method: "POST" });
      if (!res.ok) throw new Error(`Indexing failed: ${res.status}`);
      const data = await res.json();
      setIndexMessage(
        `Indexed ${data.n_documents} documents (${data.n_chunks} chunks) for experiment "${data.experiment}".`
      );
      await refreshStatus();
    } catch (e) {
      setIndexMessage(e instanceof Error ? e.message : "Indexing failed");
    } finally {
      setIndexing(false);
    }
  }

  async function uploadQuestions(file: File) {
    setQuestionsUploading(true);
    setQuestionsError(null);
    setQuestionsResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/upload/questions`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Upload failed: ${res.status}`);
      }
      const data = await res.json();
      setQuestionsResult({ n_questions: data.n_questions, filename: data.filename });
    } catch (e) {
      setQuestionsError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setQuestionsUploading(false);
    }
  }

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.length) uploadFiles(e.dataTransfer.files);
  };

  const statusEntries = Object.entries(statusFiles);

  return (
    <div className="container mx-auto p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold">Bring Your Own Corpus</h1>
        <p className="text-muted-foreground">
          Upload your own documents and questions instead of the bundled EnterpriseRAG-Bench corpus.
        </p>
      </div>

      <Tabs defaultValue="documents">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="questions">Questions</TabsTrigger>
        </TabsList>

        <TabsContent value="documents" className="space-y-4 mt-4">
          <Card
            className={`border-2 border-dashed transition-colors ${
              dragActive ? "border-primary bg-primary/5" : "border-border"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <CardContent className="pt-10 pb-10 flex flex-col items-center gap-3 text-center">
              <UploadCloud className="h-10 w-10 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Drag & drop .txt, .md, .pdf, .docx, .csv, or .html files here
              </p>
              <Button variant="outline" size="sm" onClick={() => docsInputRef.current?.click()} disabled={uploading}>
                {uploading ? "Uploading..." : "Or browse files"}
              </Button>
              <input
                ref={docsInputRef}
                type="file"
                multiple
                className="hidden"
                accept=".txt,.md,.pdf,.docx,.csv,.html,.htm"
                onChange={(e) => e.target.files && uploadFiles(e.target.files)}
              />
            </CardContent>
          </Card>

          {results.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Recently uploaded</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {results.map((r, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/50 border border-border">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">{r.filename}</span>
                    </div>
                    {r.status === "error" ? (
                      <Badge variant="destructive" className="text-xs gap-1">
                        <XCircle className="h-3 w-3" /> {r.error}
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs gap-1">
                        <CheckCircle2 className="h-3 w-3" /> parsed
                      </Badge>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="pb-3 flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base">Uploaded files ({statusEntries.length})</CardTitle>
                <CardDescription>Status per file — parsed, indexed, or error.</CardDescription>
              </div>
              <Button size="sm" onClick={indexNow} disabled={indexing || statusEntries.length === 0}>
                {indexing ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Database className="h-4 w-4 mr-2" />}
                Index now
              </Button>
            </CardHeader>
            <CardContent className="space-y-2">
              {statusEntries.length === 0 && (
                <EmptyState
                  title="No files uploaded yet"
                  description="Drop files above to parse and index your own corpus."
                />
              )}
              {statusEntries.map(([fileId, entry]) => (
                <div key={fileId} className="flex items-center justify-between px-3 py-2 rounded-lg border border-border">
                  <span className="text-sm">{entry.filename}</span>
                  <Badge
                    variant={entry.status === "error" ? "destructive" : entry.status === "indexed" ? "default" : "secondary"}
                    className="text-xs"
                  >
                    {entry.status}
                  </Badge>
                </div>
              ))}
              {indexMessage && <p className="text-xs text-muted-foreground pt-2">{indexMessage}</p>}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="questions" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Upload your own Q&A golden set</CardTitle>
              <CardDescription>
                .jsonl (one {"{question, answer}"} object per line) or .csv (question,answer[,source_type,category])
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" size="sm" onClick={() => questionsInputRef.current?.click()} disabled={questionsUploading}>
                {questionsUploading ? "Uploading..." : "Choose file"}
              </Button>
              <input
                ref={questionsInputRef}
                type="file"
                className="hidden"
                accept=".jsonl,.csv"
                onChange={(e) => e.target.files?.[0] && uploadQuestions(e.target.files[0])}
              />
              {questionsResult && (
                <p className="text-sm text-emerald-600 mt-3">
                  Loaded {questionsResult.n_questions} questions from {questionsResult.filename}.
                </p>
              )}
              {questionsError && <p className="text-sm text-destructive mt-3">{questionsError}</p>}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export type Confidence = "High" | "Medium" | "Low";
export type AuthProviderName = "google" | "github" | "dev";

export interface CurrentUser {
  subject: string;
  provider: AuthProviderName;
  email?: string | null;
  name?: string | null;
  avatar_url?: string | null;
}

export interface AuthSessionResponse {
  authenticated: boolean;
  user: CurrentUser | null;
  available_providers: AuthProviderName[];
}

export interface Citation {
  doc_name: string;
  page_range: string;
  excerpt: string;
  source_relative_path?: string | null;
  source_level?: string | null;
  policy_category?: string | null;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  confidence: Confidence;
  metrics: Record<string, unknown>;
  retrieval_question: string;
  retrieval_debug: RetrievalDebug | null;
}

export interface ChatTurn {
  user: string;
  assistant: string;
}

export interface ChatResponse extends AskResponse {
  history: ChatTurn[];
}

export interface DocumentRecord {
  corpus_id?: string | null;
  doc_name: string;
  doc_format: string;
  page_count: number;
  table_count: number;
  last_ingested_at: string;
  warnings: string[];
  source_relative_path?: string | null;
  source_level?: string | null;
  policy_category?: string | null;
  effective_date?: string | null;
  drift_status?: string | null;
  doc_date?: string | null;
  doc_author?: string | null;
  doc_type?: string | null;
  doc_summary?: string | null;
}

export interface JobRecord {
  job_id: string;
  kind: string;
  filename?: string | null;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  progress?: number;
}

export interface DocumentListResponse {
  documents: DocumentRecord[];
  active_jobs: JobRecord[];
  recent_jobs: JobRecord[];
}

export interface UploadResponse {
  job_id: string;
  status: string;
  filename: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  kind: string;
  status: string;
  filename?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  report_id?: string | null;
  progress?: number;
}

export interface EvalReportSummary {
  report_id: string;
  generated_at: string;
  prompt_count: number;
  averages: Record<string, number>;
}

export interface EvalResultRow {
  question_id: string;
  question: string;
  category: string;
  result: {
    metrics: Record<string, unknown>;
    citations: string[];
  };
  scores: Record<string, number>;
  judge: {
    reasoning?: string;
    faithfulness?: number;
    relevance?: number;
  };
}

export interface EvalReportResponse {
  report_id: string;
  generated_at: string;
  prompt_count: number;
  judge_failures: number;
  averages: Record<string, number>;
  results: EvalResultRow[];
}

export interface CorpusRecord {
  id: string;
  name: string;
  source_path: string;
  active: boolean;
  status: string;
  document_count: number;
  chunk_count: number;
  last_indexed_at?: string | null;
  last_error?: string | null;
  drift_state: string;
  source_file_count: number;
  active_jobs: JobRecord[];
  recent_jobs: JobRecord[];
}

export interface CorpusListResponse {
  active_corpus_id: string;
  corpora: CorpusRecord[];
}

export interface CorpusHealthResponse {
  corpus_id: string;
  drift_state: string;
  source_path: string;
  source_file_count: number;
  generated_document_count: number;
  indexed_document_count: number;
  chunk_count: number;
  parent_chunk_count: number;
  missing_source_count: number;
  new_source_count: number;
  changed_source_count: number;
  missing_chunk_file_count: number;
  last_indexed_at?: string | null;
  missing_sources: string[];
  new_sources: string[];
  changed_sources: string[];
}

export interface SearchResult {
  chunk_id: string;
  doc_name: string;
  source_relative_path?: string | null;
  source_level?: string | null;
  policy_category?: string | null;
  page_range: string;
  excerpt: string;
  score: number;
  matched_terms: string[];
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface RetrievalCandidate {
  chunk_id: string;
  doc_name: string;
  page_start: number;
  page_end: number;
  retrieval_channels: string[];
  matched_terms: string[];
  effective_score: number;
  fusion_score: number;
  similarity_score: number;
  keyword_score: number;
  selection_reason: string;
  has_table?: boolean;
  citation?: string;
  text?: string;
  child_text?: string;
  context_order?: number;
}

export interface GroundingDebug {
  draft_answer?: string;
  grounded_answer?: string;
  unsupported_claims_removed?: string[];
  verdict?: string;
  review_raw?: string;
}

export interface RetrievalDebug {
  candidates: RetrievalCandidate[];
  retrieved?: RetrievalCandidate[];
  selected: RetrievalCandidate[];
  grounding?: GroundingDebug;
}

function resolveApiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (configured) {
    return configured;
  }
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

const API_BASE = resolveApiBase();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const started = typeof performance !== "undefined" ? performance.now() : Date.now();
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  const payload = (await response.json()) as T;
  const elapsedSeconds = Number((((typeof performance !== "undefined" ? performance.now() : Date.now()) - started) / 1000).toFixed(3));
  if (
    payload &&
    typeof payload === "object" &&
    "metrics" in payload &&
    payload.metrics &&
    typeof payload.metrics === "object"
  ) {
    (payload.metrics as Record<string, unknown>).client_runtime_seconds = elapsedSeconds;
  }
  return payload;
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.blob();
}

export const api = {
  healthcheck: () => request<{ status: string }>("/healthz"),
  getAuthSession: () => request<AuthSessionResponse>("/auth/me"),
  logout: () =>
    request<AuthSessionResponse>("/auth/logout", {
      method: "POST",
    }),
  loginUrl: (provider: AuthProviderName) => {
    const nextUrl =
      typeof window !== "undefined"
        ? `${window.location.origin}/library`
        : "http://localhost:5173/library";
    return `${API_BASE}/auth/login/${provider}?next=${encodeURIComponent(nextUrl)}`;
  },
  listDocuments: () => request<DocumentListResponse>("/api/documents"),
  listCorpora: () => request<CorpusListResponse>("/api/corpora"),
  createCorpus: (payload: { name: string; source_path: string; corpus_id?: string | null }) =>
    request<CorpusRecord>("/api/corpora", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getCorpusHealth: (corpusId: string) =>
    request<CorpusHealthResponse>(`/api/corpora/${encodeURIComponent(corpusId)}/health`),
  activateCorpus: (corpusId: string) =>
    request<CorpusRecord>(`/api/corpora/${encodeURIComponent(corpusId)}/activate`, {
      method: "POST",
    }),
  rebuildCorpus: (corpusId: string) =>
    request<JobStatusResponse>(`/api/corpora/${encodeURIComponent(corpusId)}/rebuild`, {
      method: "POST",
    }),
  resetCorpus: (corpusId: string) =>
    request<JobStatusResponse>(`/api/corpora/${encodeURIComponent(corpusId)}/reset`, {
      method: "POST",
    }),
  replaceCorpusSource: (corpusId: string, sourcePath: string) =>
    request<JobStatusResponse>(`/api/corpora/${encodeURIComponent(corpusId)}/replace-source`, {
      method: "POST",
      body: JSON.stringify({ source_path: sourcePath }),
    }),
  search: (payload: { query: string; source_level?: string | null; policy_category?: string | null; limit?: number }) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  uploadDocument: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/api/documents/upload", {
      method: "POST",
      body: form,
    });
  },
  deleteDocument: (docName: string) =>
    request<UploadResponse>(`/api/documents/${encodeURIComponent(docName)}`, {
      method: "DELETE",
    }),
  getRecommendations: () =>
    request<{ recommendations: string[] }>("/api/query/recommendations"),
  ask: (question: string, debug: boolean, groundingEnabled: boolean, after?: string) =>
    request<AskResponse>("/api/query/ask", {
      method: "POST",
      body: JSON.stringify({ question, debug, grounding_enabled: groundingEnabled, after: after || null }),
    }),
  chat: (message: string, history: ChatTurn[], debug: boolean, groundingEnabled: boolean, after?: string) =>
    request<ChatResponse>("/api/query/chat", {
      method: "POST",
      body: JSON.stringify({ message, history, debug, grounding_enabled: groundingEnabled, after: after || null }),
    }),
  runEval: () =>
    request<UploadResponse>("/api/eval/run", {
      method: "POST",
    }),
  getEvalJob: (jobId: string) => request<JobStatusResponse>(`/api/eval/jobs/${encodeURIComponent(jobId)}`),
  listEvalReports: () => request<EvalReportSummary[]>("/api/eval/reports"),
  getEvalReport: (reportId: string) => request<EvalReportResponse>(`/api/eval/reports/${encodeURIComponent(reportId)}`),
  exportPdf: (payload: { title: string; question: string; answer: string; citations: Citation[]; generated_at?: string | null }) =>
    requestBlob("/api/export/pdf", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  exportPptx: (payload: { title: string; question: string; answer: string; citations: Citation[]; generated_at?: string | null }) =>
    requestBlob("/api/export/pptx", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

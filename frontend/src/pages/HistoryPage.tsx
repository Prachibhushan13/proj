import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

export function HistoryPage() {
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => { api.get("/history").then(({ data }) => setHistory(data)).catch(() => setHistory([])).finally(() => setLoading(false)); }, []);
  return <AppShell><div className="mx-auto max-w-5xl space-y-6">
    <header className="border-b border-slate-200 pb-5"><p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">Workspace / Records</p><h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Analysis history</h1><p className="mt-2 text-sm text-slate-600">Previously completed video screening reports.</p></header>
    <section className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      {loading ? <p className="p-6 text-sm text-slate-500">Loading records…</p> : history.length ? history.map((item) => <Link key={item.id} to={`/results/${item.id}`} state={item} className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 px-5 py-4 last:border-0 hover:bg-slate-50"><span className="min-w-0"><span className="block truncate text-sm font-medium text-slate-900">{item.filename}</span><span className="mt-1 block text-xs text-slate-500">{new Date(item.createdAt).toLocaleString()}</span></span><span className="flex items-center gap-4"><span className="text-sm font-medium text-slate-800">{item.result}</span><span className="w-20 text-right text-sm tabular-nums text-slate-600">{item.confidence.toFixed(1)}%</span></span></Link>) : <div className="px-5 py-12 text-center"><p className="font-medium text-slate-800">No analysis records</p><p className="mt-1 text-sm text-slate-500">New results will appear here after a successful analysis.</p><Link to="/upload" className="mt-4 inline-block text-sm font-medium text-teal-800 underline">Analyze a video</Link></div>}
    </section>
  </div></AppShell>;
}

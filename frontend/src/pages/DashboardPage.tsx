import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, FileVideo2 } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

export function DashboardPage() {
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => { api.get("/history").then(({ data }) => setHistory(data)).catch(() => setHistory([])).finally(() => setLoading(false)); }, []);
  const latest = history[0];
  const flagged = history.filter((item) => item.result === "FAKE").length;
  const inconclusive = history.filter((item) => item.result === "INCONCLUSIVE").length;
  return <AppShell><div className="mx-auto max-w-5xl space-y-7">
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-6"><div><p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">Workspace overview</p><h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Analysis dashboard</h1><p className="mt-2 text-sm text-slate-600">Review recent video screening results.</p></div><Link to="/upload" className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800">Analyze video<ArrowUpRight size={16} /></Link></header>
    <section className="grid gap-4 sm:grid-cols-3">{[{ label: "Total analyses", value: history.length }, { label: "Flagged as deepfake", value: flagged }, { label: "Inconclusive", value: inconclusive }].map((item) => <div key={item.label} className="rounded-lg border border-slate-200 bg-white p-5"><p className="text-sm text-slate-500">{item.label}</p><p className="mt-2 text-3xl font-semibold tabular-nums text-slate-900">{item.value}</p></div>)}</section>
    {latest && <section className="rounded-lg border border-slate-200 bg-white p-5"><div className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Most recent</p><h2 className="mt-2 max-w-xl truncate font-semibold text-slate-900">{latest.filename}</h2><p className="mt-1 text-sm text-slate-500">{new Date(latest.createdAt).toLocaleString()}</p></div><Link to={`/results/${latest.id}`} state={latest} className="text-sm font-medium text-teal-800 underline">View report</Link></div><div className="mt-4 flex items-center gap-3 border-t border-slate-100 pt-4"><span className="text-sm text-slate-600">Outcome</span><span className="rounded border border-slate-200 px-2.5 py-1 text-sm font-medium text-slate-900">{latest.result}</span><span className="text-sm text-slate-500">Model score {latest.confidence.toFixed(1)}%</span></div></section>}
    <section className="overflow-hidden rounded-lg border border-slate-200 bg-white"><div className="flex items-center justify-between border-b border-slate-200 px-5 py-4"><h2 className="font-semibold text-slate-900">Recent analysis records</h2><Link to="/history" className="text-sm font-medium text-teal-800 underline">Full history</Link></div>{loading ? <p className="px-5 py-8 text-sm text-slate-500">Loading records…</p> : history.length ? history.slice(0, 5).map((item) => <Link key={item.id} to={`/results/${item.id}`} state={item} className="flex items-center justify-between gap-4 border-b border-slate-100 px-5 py-4 last:border-0 hover:bg-slate-50"><span className="flex min-w-0 items-center gap-3"><FileVideo2 size={18} className="shrink-0 text-slate-500" /><span className="min-w-0"><span className="block truncate text-sm font-medium text-slate-900">{item.filename}</span><span className="mt-1 block text-xs text-slate-500">{new Date(item.createdAt).toLocaleString()}</span></span></span><span className="shrink-0 text-sm text-slate-700">{item.result}</span></Link>) : <div className="px-5 py-10 text-center"><p className="font-medium text-slate-800">No analyses yet</p><p className="mt-1 text-sm text-slate-500">Upload a video to create your first analysis record.</p></div>}</section>
    <p className="text-xs text-slate-500">Model scores are screening signals and should not be treated as proof of authenticity.</p>
  </div></AppShell>;
}

import { ArrowUpRight, Info } from "lucide-react";
import { Link } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";

export function ComparePage() {
  return <AppShell><div className="mx-auto max-w-4xl space-y-6">
    <header className="border-b border-slate-200 pb-5"><p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">Model details</p><h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Inference configuration</h1><p className="mt-2 text-sm text-slate-600">The active application path uses one temporal classifier.</p></header>
    <section className="rounded-lg border border-slate-200 bg-white p-5"><h2 className="font-semibold text-slate-900">Configured model</h2><p className="mt-2 text-sm leading-6 text-slate-600">EfficientNet-B0 features followed by a two-layer bidirectional LSTM. It processes a 20-frame RGB sequence and returns a single video-level score. Independent CNN and LSTM models are not loaded, so no model comparison is presented.</p><div className="mt-4 flex items-start gap-3 rounded-md border border-slate-200 bg-slate-50 p-4"><Info size={18} className="mt-0.5 shrink-0 text-slate-600" /><p className="text-sm leading-5 text-slate-600">The configured checkpoint must be installed locally. When it is missing or invalid, analysis reports model unavailability instead of substituting an untrained or unrelated model.</p></div><Link to="/upload" className="mt-5 inline-flex items-center gap-2 rounded-md bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800">Analyze a video<ArrowUpRight size={16} /></Link></section>
  </div></AppShell>;
}

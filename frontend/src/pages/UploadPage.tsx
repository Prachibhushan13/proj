import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileVideo2, ShieldCheck, Upload, X } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";

const accepted = new Set(["mp4", "mov", "avi", "mkv"]);
const maxSize = 250 * 1024 * 1024;

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inconclusiveInput, setInconclusiveInput] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const preview = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const chooseFile = (candidate?: File) => {
    setError(null);
    setInconclusiveInput(false);
    if (!candidate) return;
    const extension = candidate.name.split(".").pop()?.toLowerCase();
    if (!extension || !accepted.has(extension)) {
      setFile(null);
      setError("Choose an MP4, MOV, AVI, or MKV video.");
      return;
    }
    if (candidate.size > maxSize) {
      setFile(null);
      setError("This file is larger than the 250 MB upload limit.");
      return;
    }
    setFile(candidate);
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setInconclusiveInput(false);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await api.post("/predict", form);
      if (!(["trained_cnn_lstm_keras", "trained_cnn_lstm_onnx"] as string[]).includes(data.modelMode)) {
        setError(`The connected backend returned an unsupported model mode “${data.modelMode ?? "unknown"}”. Restart FastAPI from this project so it loads the configured trained CNN-LSTM checkpoint, then upload again.`);
        return;
      }
      navigate(`/results/${data.id}`, { state: data });
    } catch (failure: any) {
      const status = failure.response?.status;
      const detail = failure.response?.data?.detail;
      if (status === 422 && typeof detail === "string" && detail.toLowerCase().includes("inconclusive")) {
        setInconclusiveInput(true);
        setError(detail);
      } else if (status === 503) {
        setError(typeof detail === "string" ? detail : "The trained model is unavailable. Add its checkpoint and try again.");
      } else if (status === 422) {
        setError(typeof detail === "string" ? detail : "The video could not be decoded. Try a valid MP4, MOV, AVI, or MKV file.");
      } else if (status === 401) {
        setError("Your session has expired. Sign in again, then retry the upload.");
      } else {
        setError(typeof detail === "string" ? detail : "Analysis failed. Check the connection and try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl space-y-7">
        <header className="border-b border-slate-200 pb-6">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">Media integrity / Analysis</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Analyze a video</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">Upload a video for a screening signal from the configured temporal classifier. The result is not proof of authenticity.</p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(260px,0.65fr)]">
          <section className="space-y-5">
            <div
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}
              className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors ${dragging ? "border-teal-700 bg-teal-50" : "border-slate-300 bg-white"}`}
            >
              <div className="mx-auto grid h-12 w-12 place-items-center rounded-full bg-slate-100 text-slate-700"><Upload size={21} /></div>
              <h2 className="mt-4 font-semibold text-slate-900">Drop a video file here</h2>
              <p className="mt-1 text-sm text-slate-500">MP4, MOV, AVI, or MKV · up to 250 MB</p>
              <input ref={inputRef} type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mkv" className="sr-only" onChange={(event) => chooseFile(event.target.files?.[0])} />
              <button type="button" onClick={() => inputRef.current?.click()} className="mt-5 rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50">Browse files</button>
            </div>

            {error && <div role="alert" className={`rounded-md border px-4 py-3 text-sm ${inconclusiveInput ? "border-amber-300 bg-amber-50 text-amber-950" : "border-rose-300 bg-rose-50 text-rose-900"}`}><span className="font-semibold">{inconclusiveInput ? "Inconclusive:" : "Unable to analyze:"}</span> {error}</div>}

            {file && preview && <section className="overflow-hidden rounded-lg border border-slate-200 bg-white">
              <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
                <div className="flex min-w-0 items-center gap-3"><FileVideo2 size={19} className="shrink-0 text-teal-700" /><div className="min-w-0"><p className="truncate text-sm font-medium text-slate-900">{file.name}</p><p className="text-xs text-slate-500">{(file.size / 1048576).toFixed(1)} MB</p></div></div>
                <button type="button" aria-label="Remove selected video" onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }} className="rounded p-2 text-slate-500 hover:bg-slate-100"><X size={17} /></button>
              </div>
              <video controls preload="metadata" src={preview} className="aspect-video max-h-[420px] w-full bg-slate-950 object-contain" />
              <div className="flex items-center justify-between gap-4 p-4">
                <p className="text-xs leading-5 text-slate-500">Video is sampled into an ordered frame sequence for analysis.</p>
                <button disabled={loading} type="button" onClick={submit} className="shrink-0 rounded-md bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800 disabled:cursor-wait disabled:opacity-60">{loading ? "Analyzing…" : "Start analysis"}</button>
              </div>
              {loading && <div aria-live="polite" className="border-t border-slate-200 px-4 py-3"><div className="h-1 overflow-hidden rounded bg-slate-100"><div className="analysis-progress h-full w-1/3 rounded bg-teal-700" /></div><p className="mt-2 text-xs text-slate-600">Uploading and analyzing video. This may take a moment.</p></div>}
            </section>}
          </section>

          <aside className="h-fit rounded-lg border border-slate-200 bg-white p-5">
            <div className="flex items-center gap-2 text-slate-900"><ShieldCheck size={18} className="text-teal-700" /><h2 className="font-semibold">Analysis notes</h2></div>
            <ul className="mt-4 space-y-3 text-sm leading-5 text-slate-600">
              <li>Uses the configured trained CNN-LSTM checkpoint and reports its video-level model scores.</li>
              <li>A result score is a model signal, not a guarantee of authenticity.</li>
              <li>Frame-level evidence appears only when the model produces frame-level outputs.</li>
              <li>If the model is missing or the video cannot be decoded, analysis stops with an explanation.</li>
            </ul>
          </aside>
        </div>
      </div>
    </AppShell>
  );
}

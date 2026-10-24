import { Link, NavLink } from "react-router-dom";
import { Activity, FileClock, LayoutDashboard, LogOut, ShieldCheck, UploadCloud, SplitSquareHorizontal } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/upload", label: "Analyze video", icon: UploadCloud },
  { to: "/compare", label: "Model compare", icon: SplitSquareHorizontal },
  { to: "/analytics", label: "Analytics", icon: Activity },
  { to: "/history", label: "History", icon: FileClock },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const { logout, user } = useAuth();
  return <div className="min-h-screen bg-slate-50 text-slate-900">
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between px-5 py-3 lg:px-8">
        <Link to="/dashboard" className="flex items-center gap-2.5 font-semibold tracking-tight"><span className="grid h-8 w-8 place-items-center rounded-md bg-slate-900 text-white"><ShieldCheck size={18} /></span><span>Media Integrity Lab</span></Link>
        <div className="flex items-center gap-4"><span className="hidden text-sm text-slate-600 sm:block">{user?.name}</span><button type="button" onClick={logout} className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"><LogOut size={15} />Sign out</button></div>
      </div>
    </header>
    <div className="mx-auto flex max-w-[1440px] gap-8 px-5 py-7 lg:px-8">
      <aside className="hidden w-56 shrink-0 lg:block"><p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Workspace</p><nav className="space-y-1">{navItems.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} className={({ isActive }) => `flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors ${isActive ? "bg-teal-50 text-teal-900" : "text-slate-600 hover:bg-white hover:text-slate-900"}`}><Icon size={17} />{label}</NavLink>)}</nav>{user?.role === "admin" && <Link to="/admin" className="mt-1 flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium text-slate-600 hover:bg-white"><ShieldCheck size={17} />Administration</Link>}<p className="mt-8 border-t border-slate-200 px-3 pt-4 text-xs leading-5 text-slate-500">Automated analysis is a screening aid. Verify findings against original media.</p></aside>
      <main className="min-w-0 flex-1">{children}</main>
    </div>
  </div>;
}

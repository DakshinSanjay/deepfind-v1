import { useState, useEffect } from "react";
import {
  Search,
  Settings,
  Image as ImageIcon,
  FileText,
  Music,
  Video,
  File,
  X // NEW: Close icon for modal
} from "lucide-react";

interface SearchResult {
  file_name: string;
  file_path: string;
  file_type: string;
  snippet: string;
  score: number;
  match_type: string;
}

interface SystemStatus {
  indexed_files: number;
  tier2_queue: number;
  cpu_percent: number;
  ram_percent: number;
  on_battery: boolean;
  is_paused: boolean;
}

const FILTERS = [
  { id: "images", label: "Images", icon: ImageIcon },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "audio", label: "Audio", icon: Music },
  { id: "video", label: "Video", icon: Video },
];

const iconFor = (type: string) => {
  if (["pdf", "doc", "docx", "xlsx", "pptx", "txt", "md"].includes(type)) return FileText;
  if (["png", "jpg", "jpeg", "webp", "tiff", "bmp", "image"].includes(type)) return ImageIcon;
  if (["mp3", "wav", "m4a"].includes(type)) return Music;
  if (["mp4", "mkv", "mov"].includes(type)) return Video;
  return File;
};

export default function App() {
  const [query, setQuery] = useState("receipt");
  const [activeFilters, setActiveFilters] = useState<string[]>(["images", "documents"]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);

  // NEW: Settings Modal State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [cpuLimit, setCpuLimit] = useState(50);
  const [ramLimit, setRamLimit] = useState(75);
  const [pauseBattery, setPauseBattery] = useState(true);
  const [watchPath, setWatchPath] = useState("~/Downloads");

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setSearchError(null);
      setIsSearching(false);
      return;
    }
    if (activeFilters.length === 0) {
      setResults([]);
      setSearchError(null);
      setIsSearching(false);
      return;
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(async () => {
      setIsSearching(true);
      setSearchError(null);
      try {
        const params = new URLSearchParams({ q: query, filters: activeFilters.join(",") });
        const res = await fetch(`http://localhost:8000/api/search?${params}`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Search failed (${res.status})`);
        const data: SearchResult[] = await res.json();
        setResults(data);
      } catch (err) {
        if (!controller.signal.aborted) {
          setSearchError("DeepFind could not reach the local search service.");
          setResults([]);
        }
      } finally {
        if (!controller.signal.aborted) setIsSearching(false);
      }
    }, 250);
    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, [query, activeFilters]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/status");
        const data: SystemStatus = await res.json();
        setSysStatus(data);
      } catch (err) {}
    };
    fetchStatus();
    const intervalId = setInterval(fetchStatus, 3000); 
    return () => clearInterval(intervalId);
  }, []);

  const toggleFilter = (id: string) => {
    setActiveFilters((prev) =>
      prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id]
    );
  };

  const handleReveal = async (filePath: string) => {
    try {
      await fetch("http://localhost:8000/api/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath }),
      });
    } catch (err) {}
  };

  // NEW: Save Settings to Backend
  const handleSaveSettings = async () => {
    try {
      await fetch("http://localhost:8000/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          watched_paths: [watchPath],
          cpu_limit: cpuLimit,
          ram_limit: ramLimit,
          pause_on_battery: pauseBattery
        }),
      });
      setIsSettingsOpen(false);
    } catch (err) {
      console.error("Failed to save settings", err);
    }
  };

  return (
    <div className="df-root">
      <style>{`
        .df-root { --bg: #ffffff; --navy-900: #161f52; --navy-800: #1c2762; --navy-700: #26326f; --navy-600: #2f3c85; --teal: #2bb6c2; --teal-soft: #8fdee4; --icon-bg: #e3ebfb; --text-muted: #6b7280; --lavender: #aab1de; --lavender-dim: #7d84b8; font-family: 'DejaVu Sans', sans-serif; background: var(--bg); min-height: 100vh; color: var(--navy-900); display: flex; flex-direction: column; }
        .df-topbar { display: flex; align-items: center; justify-content: space-between; padding: 22px 44px 0 44px; }
        .df-wordmark { font-family: Georgia, serif; font-weight: 700; font-size: 20px; color: var(--navy-900); }
        .df-status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); }
        .df-status-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--teal); transition: background 0.3s; }
        .df-settings { color: var(--text-muted); margin-left: 18px; cursor: pointer; transition: color 0.2s; }
        .df-settings:hover { color: var(--navy-900); }
        .df-hero { padding: 40px 44px 28px 44px; max-width: 720px; }
        .df-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--teal); margin-bottom: 14px; }
        .df-headline { font-family: Georgia, serif; font-weight: 600; font-size: 34px; line-height: 1.18; color: var(--navy-900); margin-bottom: 12px; }
        .df-subtext { font-size: 15px; color: var(--text-muted); line-height: 1.5; max-width: 460px; }
        .df-filters { display: flex; gap: 10px; padding: 4px 44px 32px 44px; flex-wrap: wrap; }
        .df-filter { display: flex; align-items: center; gap: 9px; padding: 8px 16px 8px 8px; border-radius: 999px; border: 1px solid #e7e9f2; background: #fff; cursor: pointer; font: inherit; }
        .df-filter[data-active="true"] { border-color: var(--navy-900); background: #f4f6ff; }
        .df-filter-icon { width: 26px; height: 26px; border-radius: 999px; background: var(--icon-bg); display: flex; align-items: center; justify-content: center; color: var(--navy-900); }
        .df-filter-label { font-size: 13.5px; font-weight: 600; color: var(--navy-900); }
        .df-panel { background: linear-gradient(180deg, var(--navy-800), var(--navy-900)); margin: 0 44px 44px 44px; border-radius: 22px; padding: 30px 30px 20px 30px; flex: 1; display: flex; flex-direction: column; }
        .df-searchbar { display: flex; align-items: center; gap: 12px; background: var(--navy-700); border-radius: 14px; padding: 15px 18px; margin-bottom: 22px; }
        .df-searchbar input { flex: 1; background: transparent; border: none; outline: none; color: #fff; font-size: 15px; font-style: italic; }
        .df-searchbar input::placeholder { color: var(--lavender-dim); font-style: italic; }
        .df-searchbar svg { color: var(--teal-soft); }
        .df-results { display: flex; flex-direction: column; gap: 10px; flex: 1; }
        .df-empty { color: var(--lavender); font-size: 14px; padding: 18px 4px; }
        .df-row { display: flex; align-items: center; gap: 14px; background: var(--navy-600); border-radius: 12px; padding: 14px 16px; cursor: pointer; }
        .df-row:hover { background: #37469a; }
        .df-row-icon { width: 36px; height: 36px; border-radius: 10px; background: rgba(255, 255, 255, 0.08); display: flex; align-items: center; justify-content: center; color: var(--teal-soft); }
        .df-row-main { flex: 1; min-width: 0; }
        .df-row-name { font-size: 14.5px; font-weight: 600; color: #fff; }
        .df-row-path { font-size: 12px; color: var(--lavender-dim); margin-top: 1px; }
        .df-row-snippet { font-size: 12.5px; font-style: italic; color: var(--lavender); margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .df-match { font-size: 12px; font-weight: 700; color: var(--navy-900); background: var(--teal-soft); padding: 5px 11px; border-radius: 999px; }
        .df-panel-footer { display: flex; align-items: center; gap: 12px; padding: 18px 4px 0px 4px; font-size: 12px; color: var(--lavender-dim); margin-top: auto; }
        .df-divider { width: 1px; height: 11px; background: var(--lavender-dim); opacity: 0.5; }
        
        /* NEW: Modal Styling */
        .df-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 50; }
        .df-modal { background: #fff; width: 440px; border-radius: 16px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .df-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; font-weight: 600; font-size: 18px; }
        .df-modal-close { cursor: pointer; color: var(--text-muted); }
        .df-modal-field { margin-bottom: 18px; }
        .df-modal-label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--navy-900); }
        .df-modal-input { width: 100%; padding: 10px; border: 1px solid #e7e9f2; border-radius: 8px; font-family: 'DejaVu Sans', sans-serif; font-size: 14px; box-sizing: border-box; }
        .df-modal-slider { width: 100%; margin-top: 4px; }
        .df-modal-btn { background: var(--navy-900); color: #fff; border: none; padding: 12px; width: 100%; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 10px; }
      `}</style>

      {/* NEW: Settings Modal Overlay */}
      {isSettingsOpen && (
        <div className="df-modal-overlay">
          <div className="df-modal">
            <div className="df-modal-header">
              Engine Throttling & Limits
              <X size={20} className="df-modal-close" onClick={() => setIsSettingsOpen(false)} />
            </div>
            
            <div className="df-modal-field">
              <label className="df-modal-label">Watched Folder Path</label>
              <input type="text" className="df-modal-input" value={watchPath} onChange={(e) => setWatchPath(e.target.value)} />
            </div>

            <div className="df-modal-field">
              <label className="df-modal-label">Max CPU Usage: {cpuLimit}%</label>
              <input type="range" min="10" max="100" value={cpuLimit} className="df-modal-slider" onChange={(e) => setCpuLimit(Number(e.target.value))} />
            </div>

            <div className="df-modal-field">
              <label className="df-modal-label">Max RAM Usage: {ramLimit}%</label>
              <input type="range" min="10" max="100" value={ramLimit} className="df-modal-slider" onChange={(e) => setRamLimit(Number(e.target.value))} />
            </div>

            <div className="df-modal-field" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" id="battery" checked={pauseBattery} onChange={(e) => setPauseBattery(e.target.checked)} />
              <label htmlFor="battery" style={{ fontSize: '14px' }}>Pause heavy indexing on battery power</label>
            </div>

            <button className="df-modal-btn" onClick={handleSaveSettings}>Save Configuration</button>
          </div>
        </div>
      )}

      <div className="df-topbar">
        <div className="df-wordmark">DeepFind</div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <div className="df-status">
            <span className="df-status-dot" style={{ background: sysStatus?.is_paused ? 'orange' : 'var(--teal)' }} />
            {sysStatus ? (sysStatus.is_paused ? "Indexing Paused" : "Index Active") : "Connecting..."}
          </div>
          {/* NEW: Click handler to open the modal */}
          <Settings size={17} className="df-settings" onClick={() => setIsSettingsOpen(true)} />
        </div>
      </div>

      <div className="df-hero">
        <div className="df-eyebrow">Private and offline</div>
        <div className="df-headline">Search everything on your laptop, by what it means.</div>
        <div className="df-subtext">A background service reads the content inside your files and makes it searchable in plain language.</div>
      </div>

      <div className="df-filters">
        {FILTERS.map((f) => {
          const Icon = f.icon;
          const isActive = activeFilters.includes(f.id);
          return (
            <button key={f.id} type="button" className="df-filter" data-active={isActive} onClick={() => toggleFilter(f.id)}>
              <span className="df-filter-icon"><Icon size={13} /></span>
              <span className="df-filter-label">{f.label}</span>
            </button>
          );
        })}
      </div>

      <div className="df-panel">
        <div className="df-searchbar">
          <Search size={17} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by what it means..." />
        </div>

        <div className="df-results">
          {isSearching && <div className="df-empty">Searching your indexed files...</div>}
          {!isSearching && searchError && <div className="df-empty" role="alert">{searchError}</div>}
          {!isSearching && !searchError && results.length === 0 && (
            <div className="df-empty">
              {activeFilters.length === 0
                ? "Choose at least one file type to search."
                : query.trim()
                  ? "No matching files found."
                  : "Type to search your indexed files."}
            </div>
          )}
          {!isSearching && results.map((r, idx) => {
            const Icon = iconFor(r.file_type);
            return (
              <div className="df-row" key={idx} onClick={() => handleReveal(r.file_path)}>
                <span className="df-row-icon"><Icon size={16} /></span>
                <div className="df-row-main">
                  <div className="df-row-name">{r.file_name}</div>
                  <div className="df-row-path">{r.file_path}</div>
                  <div className="df-row-snippet">{r.snippet}</div>
                </div>
                <span className="df-match">{Math.round(r.score * 100)}% {r.match_type}</span>
              </div>
            );
          })}
        </div>

        <div className="df-panel-footer">
          <span>{sysStatus ? sysStatus.indexed_files.toLocaleString() : 0} files indexed</span>
          <span className="df-divider" />
          <span>{sysStatus ? sysStatus.tier2_queue : 0} in Tier 2 queue (OCR)</span>
          <span className="df-divider" />
          <span>CPU: {sysStatus?.cpu_percent || 0}%</span>
          <span className="df-divider" />
          <span>RAM: {sysStatus?.ram_percent || 0}%</span>
        </div>
      </div>
    </div>
  );
}
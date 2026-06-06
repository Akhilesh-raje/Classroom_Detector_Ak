import React, { useRef, useEffect, useState } from 'react';
import { api } from '../api';
import { 
  Play, 
  Pause, 
  Upload, 
  Terminal, 
  User, 
  BarChart3, 
  AlertCircle, 
  CheckCircle2, 
  Loader2,
  FileVideo,
  Monitor
} from 'lucide-react';

const EMOTION_COLORS = {
  happy: '#10b981',     // emerald
  sad: '#3b82f6',       // blue
  angry: '#ef4444',     // red
  surprised: '#f59e0b', // amber
  fearful: '#8b5cf6',   // violet
  disgusted: '#14b8a6', // teal
  neutral: '#64748b'    // slate
};

export default function TestVideo() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const scrollRef = useRef(null);
  
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [currentFaces, setCurrentFaces] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState(null);
  const [serverVideos, setServerVideos] = useState([]);

  // Fetch server-side test videos on mount
  useEffect(() => {
    api.getVideoList()
      .then(res => setServerVideos(res.videos || []))
      .catch(err => addLog(`Failed to fetch server videos: ${err.message}`, 'error'));
  }, []);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => {
      const newLogs = [...prev, { 
        id: Date.now() + Math.random(), 
        msg, 
        type, 
        time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) 
      }];
      return newLogs.slice(-100);
    });
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      addLog(`Selected local video: ${file.name}`);
      resetState();
    }
  };

  const selectServerVideo = (video) => {
    setVideoFile({ serverPath: video.path, name: video.filename });
    setVideoUrl(`http://localhost:8000/test-videos/${encodeURIComponent(video.filename)}`);
    addLog(`Selected server video: ${video.filename}`);
    resetState();
  };

  const resetState = () => {
    setCurrentFaces([]);
    setMetadata(null);
    setProgress(0);
    setStats(null);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const startAnalysis = async () => {
    if (!videoFile) return;
    
    setIsProcessing(true);
    addLog('Starting video analysis (Python Backend)...', 'success');
    
    try {
      let response;
      if (videoFile.serverPath) {
        response = await api.analyzeVideoPath(videoFile.serverPath);
      } else {
        response = await api.analyzeVideo(videoFile);
      }

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep last incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            handleBackendEvent(data);
          } catch (e) {
            console.error("Parse error", e, line);
          }
        }
      }
      addLog('Analysis finished', 'success');
    } catch (err) {
      addLog(`Error: ${err.message}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBackendEvent = (data) => {
    if (data.type === 'metadata') {
      setMetadata(data);
      addLog(`Video metadata: ${data.width}x${data.height}, ${data.duration.toFixed(1)}s`);
    } else if (data.type === 'frame') {
      setCurrentFaces(data.faces);
      if (metadata) {
        setProgress((data.frameIndex / metadata.totalFrames) * 100);
      }
      
      // Update canvas overlay
      drawOverlays(data.faces);

      // Sync video playback if possible (rough sync)
      if (videoRef.current && Math.abs(videoRef.current.currentTime - data.timestamp) > 0.5) {
        // videoRef.current.currentTime = data.timestamp;
      }
    } else if (data.type === 'complete') {
      setStats(data);
      addLog(`Analysis complete. Processed ${data.framesProcessed} frames.`, 'success');
    }
  };

  const drawOverlays = (faces) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;
    
    // Ensure canvas matches display size
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!metadata) return;

    const scaleX = displayWidth / metadata.width;
    const scaleY = displayHeight / metadata.height;

    faces.forEach((face) => {
      const [x1, y1, x2, y2] = face.bbox;
      const rx1 = x1 * scaleX;
      const ry1 = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      // Draw Box
      const color = EMOTION_COLORS[face.emotion] || '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(rx1, ry1, rw, rh);

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 12px Inter, sans-serif';
      const label = `${face.emotion.toUpperCase()} (${face.attentionScore}%)`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(rx1, ry1 - 22, textWidth + 10, 22);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, rx1 + 5, ry1 - 7);
    });
  };

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Smart Video Analysis</h1>
          <p className="text-slate-400 mt-1">Deep facial expression and attention analysis via Python Backend</p>
        </div>
        <div className="flex gap-2">
          <label className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg cursor-pointer transition-colors border border-slate-700">
            <Upload size={18} />
            <span>Upload Video</span>
            <input type="file" hidden accept="video/*" onChange={handleFileChange} />
          </label>
          <button 
            onClick={startAnalysis}
            disabled={!videoFile || isProcessing}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-semibold transition-all ${
              !videoFile || isProcessing 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20'
            }`}
          >
            {isProcessing ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            <span>{isProcessing ? 'Analyzing...' : 'Start Analysis'}</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Video Area */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          <div className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
            {videoUrl ? (
              <>
                <video 
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-full object-contain"
                  controls
                  onPlay={() => addLog('Video playback started')}
                  onPause={() => addLog('Video playback paused')}
                />
                <canvas 
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                />
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-4">
                <FileVideo size={64} strokeWidth={1} />
                <p>Upload a video to begin deep analysis</p>
              </div>
            )}
            
            {isProcessing && (
              <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1 bg-indigo-600/90 text-white rounded-full text-xs font-bold animate-pulse">
                <div className="w-2 h-2 bg-white rounded-full" />
                LIVE PROCESSING
              </div>
            )}
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">Analysis Progress</span>
              <span className="font-mono text-indigo-400">{progress.toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-indigo-500 transition-all duration-300 shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
                style={{ width: `${progress}%` }} 
              />
            </div>
          </div>

          {/* Terminal Console */}
          <div className="bg-[#0f172a] rounded-xl border border-slate-800 overflow-hidden shadow-lg flex flex-col h-[300px]">
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 border-b border-slate-800">
              <Terminal size={14} className="text-indigo-400" />
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Analysis Terminal</span>
            </div>
            <div 
              ref={scrollRef}
              className="p-4 font-mono text-[11px] leading-relaxed overflow-y-auto flex-1 scrollbar-thin"
            >
              {logs.map((log) => (
                <div key={log.id} className="flex gap-3 mb-1 group">
                  <span className="text-slate-500 shrink-0">[{log.time}]</span>
                  <span className={`${
                    log.type === 'error' ? 'text-red-400' : 
                    log.type === 'success' ? 'text-emerald-400' : 
                    'text-slate-300'
                  }`}>
                    {log.type === 'error' ? '✖ ' : log.type === 'success' ? '✔ ' : 'i '} 
                    {log.msg}
                  </span>
                </div>
              ))}
              {logs.length === 0 && <div className="text-slate-600 italic">Waiting for process start...</div>}
            </div>
          </div>
        </div>

        {/* Sidebar Analysis Details */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          {/* Server Videos */}
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800">
            <h3 className="flex items-center gap-2 text-sm font-bold text-slate-300 mb-4 uppercase tracking-wider">
              <Monitor size={16} className="text-indigo-400" />
              Server Test Videos
            </h3>
            <div className="flex flex-col gap-2 max-h-[150px] overflow-y-auto pr-2">
              {serverVideos.map((v, i) => (
                <button
                  key={i}
                  onClick={() => selectServerVideo(v)}
                  className="text-left px-3 py-2 bg-slate-800/50 hover:bg-slate-700 rounded-lg text-xs text-slate-300 transition-colors border border-transparent hover:border-slate-600 truncate"
                >
                  {v.filename}
                </button>
              ))}
            </div>
          </div>

          {/* Active Faces */}
          <div className="flex-1 bg-slate-900/80 p-5 rounded-2xl border border-slate-800 flex flex-col gap-4 overflow-hidden">
            <h3 className="flex items-center gap-2 text-sm font-bold text-slate-300 uppercase tracking-wider">
              <User size={16} className="text-indigo-400" />
              Detected Faces ({currentFaces.length})
            </h3>
            
            <div className="flex-1 overflow-y-auto pr-2 flex flex-col gap-4 scrollbar-thin">
              {currentFaces.length > 0 ? currentFaces.map((face, idx) => (
                <div key={idx} className="bg-slate-800/40 p-4 rounded-xl border border-slate-700 hover:border-slate-600 transition-all group">
                  <div className="flex gap-4 mb-4">
                    {face.thumbnail ? (
                      <img 
                        src={`data:image/jpeg;base64,${face.thumbnail}`} 
                        className="w-16 h-16 rounded-lg object-cover ring-2 ring-indigo-500/20"
                        alt="Face"
                      />
                    ) : (
                      <div className="w-16 h-16 bg-slate-800 rounded-lg flex items-center justify-center text-slate-600">
                        <User size={24} />
                      </div>
                    )}
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-sm">Face #{idx + 1}</span>
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                          face.attentionScore > 70 ? 'bg-emerald-500/10 text-emerald-400' :
                          face.attentionScore > 40 ? 'bg-amber-500/10 text-amber-400' :
                          'bg-red-500/10 text-red-400'
                        }`}>
                          {face.attentionScore}% Attentive
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 flex items-center gap-2 italic">
                        {face.activity.replace('_', ' ')}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {Object.entries(face.expressions).sort((a,b) => b[1] - a[1]).slice(0, 3).map(([emo, val]) => (
                      <div key={emo}>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="capitalize text-slate-400">{emo}</span>
                          <span className="font-mono text-slate-500">{(val * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full transition-all duration-500" 
                            style={{ 
                              width: `${val * 100}%`, 
                              backgroundColor: EMOTION_COLORS[emo] || '#fff'
                            }} 
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )) : (
                <div className="flex flex-col items-center justify-center py-12 text-slate-600 italic text-sm">
                  No faces detected in current frame
                </div>
              )}
            </div>
          </div>

          {/* Stats Card */}
          {stats && (
            <div className="bg-indigo-600/10 p-5 rounded-2xl border border-indigo-500/20 animate-in zoom-in-95 duration-500">
              <h3 className="flex items-center gap-2 text-sm font-bold text-indigo-400 uppercase tracking-wider mb-4">
                <BarChart3 size={16} />
                Analysis Results
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">{stats.totalFaceDetections}</div>
                  <div className="text-[10px] text-indigo-300/60 uppercase">Detections</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">{stats.elapsedSeconds}s</div>
                  <div className="text-[10px] text-indigo-300/60 uppercase">Time</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

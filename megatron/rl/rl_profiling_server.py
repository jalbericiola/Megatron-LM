# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
RL Profiling Dashboard Server

A simple web server to browse and compare profiling data from multiple runs.

Usage:
    # Start the server pointing to a runs directory
    python -m megatron.rl.rl_profiling_server --runs-dir /path/to/runs --port 8080

    # Then open http://localhost:8080 in your browser
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import html as html_lib

def find_all_runs(runs_dir: str) -> List[Dict[str, Any]]:
    """Find all runs with profiling data."""
    runs = []
    runs_path = Path(runs_dir)
    
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        profile_dirs = [run_dir / "checkpoints" / "profiles", run_dir / "profiles"]
        
        for profile_dir in profile_dirs:
            if profile_dir.exists():
                jsonl_files = list(profile_dir.glob("profile_*.jsonl"))
                csv_files = list(profile_dir.glob("summary_*.csv"))
                
                if jsonl_files or csv_files:
                    latest_jsonl = sorted(jsonl_files)[-1] if jsonl_files else None
                    latest_csv = sorted(csv_files)[-1] if csv_files else None
                    num_iterations = 0
                    if latest_jsonl:
                        with open(latest_jsonl) as f:
                            num_iterations = sum(1 for _ in f)
                    
                    runs.append({
                        "name": run_dir.name,
                        "path": str(run_dir),
                        "profile_dir": str(profile_dir),
                        "jsonl": str(latest_jsonl) if latest_jsonl else None,
                        "csv": str(latest_csv) if latest_csv else None,
                        "num_iterations": num_iterations,
                        "modified": datetime.fromtimestamp(
                            (latest_jsonl or latest_csv).stat().st_mtime
                        ).isoformat() if (latest_jsonl or latest_csv) else None,
                    })
                break
    
    runs.sort(key=lambda x: x.get("modified", ""), reverse=True)
    return runs

def load_jsonl(path: str, last_n: int = 100) -> List[Dict]:
    profiles = []
    with open(path) as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles[-last_n:]

def load_csv_summary(path: str) -> List[Dict]:
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def generate_dashboard_html(runs: List[Dict]) -> str:
    runs_html = ""
    for run in runs:
        runs_html += f'''
        <tr onclick="window.location='/run?name={html_lib.escape(run["name"])}';" style="cursor: pointer;">
            <td><strong>{html_lib.escape(run["name"])}</strong></td>
            <td>{run["num_iterations"]}</td>
            <td>{run.get("modified", "N/A")[:19] if run.get("modified") else "N/A"}</td>
            <td><a href="/run?name={html_lib.escape(run["name"])}">View</a></td>
        </tr>'''
    
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>RL Profiling Dashboard</title>
<style>
body {{ font-family: system-ui; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; }}
th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #30363d; }}
th {{ background: #30363d; }}
tr:hover {{ background: rgba(88,166,255,0.1); }}
a {{ color: #58a6ff; }}
</style></head>
<body><div class="container">
<h1>🚀 RL Profiling Dashboard</h1>
<table><thead><tr><th>Run Name</th><th>Iterations</th><th>Last Updated</th><th>Actions</th></tr></thead>
<tbody>{runs_html}</tbody></table>
</div></body></html>'''

def generate_run_html(run: Dict, profiles: List[Dict], summary: List[Dict]) -> str:
    iterations = [p.get("iteration", 0) for p in profiles]
    elapsed = [p.get("elapsed_time_ms", 0) / 1000 for p in profiles]
    avg_time = sum(elapsed) / len(elapsed) if elapsed else 0
    
    summary_html = ""
    for row in summary[:15]:
        timer = row.get("timer_name", "")
        mean = float(row.get("mean_ms", 0))
        p95 = float(row.get("p95_ms", 0))
        summary_html += f"<tr><td>{html_lib.escape(timer)}</td><td>{mean:.1f}</td><td>{p95:.1f}</td></tr>"
    
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{html_lib.escape(run["name"])}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: system-ui; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
a {{ color: #58a6ff; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #30363d; }}
.stats {{ display: flex; gap: 1rem; }}
.stat {{ background: #161b22; padding: 1rem; border-radius: 8px; flex: 1; text-align: center; }}
.stat-value {{ font-size: 2rem; color: #58a6ff; }}
</style></head>
<body><div class="container">
<a href="/">← Back</a>
<h1>{html_lib.escape(run["name"])}</h1>
<div class="stats">
<div class="stat"><div class="stat-value">{run["num_iterations"]}</div><div>Iterations</div></div>
<div class="stat"><div class="stat-value">{avg_time:.1f}s</div><div>Avg Time</div></div>
</div>
<div class="card"><h2>Iteration Time</h2><canvas id="chart" height="100"></canvas></div>
<div class="card"><h2>Timer Summary</h2><table><tr><th>Timer</th><th>Mean (ms)</th><th>P95 (ms)</th></tr>{summary_html}</table></div>
</div>
<script>
new Chart(document.getElementById("chart"), {{
  type: "line",
  data: {{ labels: {json.dumps(iterations)}, datasets: [{{ label: "Time (s)", data: {json.dumps(elapsed)}, borderColor: "#58a6ff", fill: false }}] }},
  options: {{ responsive: true }}
}});
</script></body></html>'''

class Handler(SimpleHTTPRequestHandler):
    runs_dir = "."
    
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        
        if parsed.path in ["/", ""]:
            runs = find_all_runs(self.runs_dir)
            self.send_html(generate_dashboard_html(runs))
        elif parsed.path == "/run":
            name = query.get("name", [""])[0]
            runs = find_all_runs(self.runs_dir)
            run = next((r for r in runs if r["name"] == name), None)
            if run:
                profiles = load_jsonl(run["jsonl"]) if run.get("jsonl") else []
                summary = load_csv_summary(run["csv"]) if run.get("csv") else []
                self.send_html(generate_run_html(run, profiles, summary))
            else:
                self.send_error(404)
        else:
            self.send_error(404)
    
    def send_html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(content.encode())
    
    def log_message(self, fmt, *args):
        print(f"[{datetime.now():%H:%M:%S}] {args[0]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", "-d", required=True)
    parser.add_argument("--port", "-p", type=int, default=8080)
    args = parser.parse_args()
    
    Handler.runs_dir = args.runs_dir
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"🚀 Dashboard at http://0.0.0.0:{args.port}")
    print(f"📁 Runs: {args.runs_dir}")
    server.serve_forever()

if __name__ == "__main__":
    main()

import os
import sys
import json
import time
import glob
import threading
import mimetypes
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import pandas as pd

from fl_simulation import execute_experiment
from generate_report import generate_report

PORT = 8000
HOST = "0.0.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for directory in [STATIC_DIR, DATA_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global simulation state
simulation_lock = threading.Lock()
simulation_stop_event = threading.Event()
simulation_thread = None

sim_state = {
    "is_running": False,
    "status_text": "Idle",
    "username": "",
    "dataset": "",
    "target_col": "",
    "task_type": "classification",
    "distribution": "iid",
    "num_clients": 5,
    "num_rounds": 5,
    "current_model": "",
    "current_strategy": "",
    "current_round": 0,
    "total_rounds": 0,
    "current_step": 0,
    "total_steps": 0,
    "live_rounds": [],
    "logs": [],
    "latest_result_file": None,
    "latest_report": None,
    "error": None,
    "start_time": 0,
    "elapsed_seconds": 0
}

def log_message(msg):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    sim_state["logs"].append(formatted)
    if len(sim_state["logs"]) > 300:
        sim_state["logs"].pop(0)

def simulation_worker(params):
    global sim_state
    sim_state["is_running"] = True
    sim_state["status_text"] = "Initializing dataset & clients..."
    sim_state["error"] = None
    sim_state["live_rounds"] = []
    sim_state["logs"] = []
    sim_state["start_time"] = time.time()
    sim_state["elapsed_seconds"] = 0
    sim_state["latest_result_file"] = None
    sim_state["latest_report"] = None
    simulation_stop_event.clear()

    dataset_path = os.path.join(DATA_DIR, params.get("dataset", "Synthetic_patient-HealthCare-Monitoring_dataset.csv"))
    if not os.path.exists(dataset_path):
        sim_state["error"] = f"Dataset not found: {dataset_path}"
        sim_state["is_running"] = False
        sim_state["status_text"] = "Error"
        log_message(f"Error: {sim_state['error']}")
        return

    models = params.get("models", ["LogisticRegression", "SimpleMLP", "DeepMLP"])
    strategies = params.get("strategies", ["FedAvg", "FedProx", "FedAdam"])
    rounds = int(params.get("num_rounds", 5))

    total_phases = len(models) + (len(strategies) - 1 if "FedAvg" in strategies else len(strategies))
    sim_state["total_steps"] = total_phases * rounds
    sim_state["current_step"] = 0

    log_message(f"Starting Federated Learning simulation for {params.get('username', 'User')}")
    log_message(f"Dataset: {os.path.basename(dataset_path)} | Clients: {params.get('num_clients', 5)} | Rounds: {rounds}")

    class Args:
        data_path = dataset_path
        target_col = params.get("target_col", "Predicted Disease")
        username = params.get("username", "Manoj")
        task_type = params.get("task_type", "classification")
        distribution = params.get("distribution", "iid")
        num_clients = int(params.get("num_clients", 5))
        num_rounds = rounds
        local_epochs = int(params.get("local_epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        lr = float(params.get("lr", 0.01))
        proximal_mu = float(params.get("proximal_mu", 0.01))
        models = params.get("models", ["LogisticRegression", "SimpleMLP", "DeepMLP"])
        strategies = params.get("strategies", ["FedAvg", "FedProx", "FedAdam"])

    args = Args()

    def progress_callback(update):
        sim_state["current_model"] = update["model"]
        sim_state["current_strategy"] = update["strategy"]
        sim_state["current_round"] = update["round"]
        sim_state["total_rounds"] = update["total_rounds"]
        sim_state["current_step"] += 1
        sim_state["elapsed_seconds"] = int(time.time() - sim_state["start_time"])
        sim_state["status_text"] = f"Training {update['model']} ({update['strategy']}) - Round {update['round']}/{update['total_rounds']}"

        metric_str = f"Acc: {update['metrics'].get('accuracy', 0):.2f}%" if args.task_type == 'classification' else f"RMSE: {update['metrics'].get('rmse', 0):.4f}"
        log_message(f"[{update['model']}][{update['strategy']}] Round {update['round']}/{update['total_rounds']} | {metric_str} | Loss: {update['metrics']['loss']:.4f}")

        sim_state["live_rounds"].append({
            "model": update["model"],
            "strategy": update["strategy"],
            "round": update["round"],
            "total_rounds": update["total_rounds"],
            "step": sim_state["current_step"],
            "metrics": update["metrics"],
            "timestamp": time.strftime("%H:%M:%S")
        })

    try:
        result_file, err = execute_experiment(args, progress_callback=progress_callback, stop_event=simulation_stop_event)
        if err:
            sim_state["error"] = err
            sim_state["status_text"] = "Simulation Failed"
            log_message(f"Failed: {err}")
        elif simulation_stop_event.is_set():
            sim_state["status_text"] = "Cancelled by user"
            log_message("Simulation was stopped by user.")
        else:
            sim_state["latest_result_file"] = result_file
            log_message(f"Experiment saved to {result_file}. Generating analytical report...")
            report_info = generate_report(result_file)
            sim_state["latest_report"] = report_info
            sim_state["status_text"] = "Completed"
            log_message(f"Analytics report ready: {report_info.get('report_file', '')}")
    except Exception as e:
        sim_state["error"] = str(e)
        sim_state["status_text"] = f"Error: {e}"
        log_message(f"Unhandled error during simulation: {e}")
    finally:
        sim_state["is_running"] = False
        sim_state["elapsed_seconds"] = int(time.time() - sim_state["start_time"])


class FLServerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching for APIs
        if self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/datasets":
            self.handle_api_datasets()
        elif path == "/api/simulations":
            self.handle_api_simulations()
        elif path == "/api/simulation/detail":
            self.handle_api_simulation_detail(query)
        elif path == "/api/reports":
            self.handle_api_reports()
        elif path == "/api/report/view":
            self.handle_api_report_view(query)
        elif path.startswith("/reports/"):
            # Serve files from reports folder
            rel_path = path[len("/reports/"):]
            full_path = os.path.join(REPORTS_DIR, rel_path)
            self.serve_file(full_path)
        elif path.startswith("/data/"):
            rel_path = path[len("/data/"):]
            full_path = os.path.join(DATA_DIR, rel_path)
            self.serve_file(full_path)
        else:
            # Serve static files
            if path in ["", "/"]:
                file_path = os.path.join(STATIC_DIR, "index.html")
            else:
                rel_path = path.lstrip("/")
                file_path = os.path.join(STATIC_DIR, rel_path)

            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.serve_file(file_path)
            else:
                # Fallback to index.html for SPA routing
                fallback = os.path.join(STATIC_DIR, "index.html")
                if os.path.exists(fallback):
                    self.serve_file(fallback)
                else:
                    self.send_error(404, "File not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/simulation/start":
            self.handle_api_simulation_start()
        elif path == "/api/simulation/stop":
            self.handle_api_simulation_stop()
        else:
            self.send_error(404, "API route not found")

    def send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, full_path):
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            self.send_error(404, "File not found")
            return
        mime_type, _ = mimetypes.guess_type(full_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            with open(full_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def handle_api_status(self):
        global sim_state
        state_copy = dict(sim_state)
        if state_copy["is_running"]:
            state_copy["elapsed_seconds"] = int(time.time() - state_copy["start_time"])
        self.send_json(state_copy)

    def handle_api_datasets(self):
        datasets = []
        for file in glob.glob(os.path.join(DATA_DIR, "*.csv")):
            fname = os.path.basename(file)
            size_mb = round(os.path.getsize(file) / (1024 * 1024), 2)
            try:
                df = pd.read_csv(file, nrows=10)
                columns = df.columns.tolist()
                preview = df.head(3).to_dict(orient="records")
                
                # Check target classes if Predicted Disease exists
                class_counts = {}
                try:
                    df_full = pd.read_csv(file, usecols=["Predicted Disease"] if "Predicted Disease" in columns else [columns[-1]])
                    target_name = "Predicted Disease" if "Predicted Disease" in columns else columns[-1]
                    class_counts = df_full[target_name].value_counts().to_dict()
                    total_rows = len(df_full)
                except Exception:
                    total_rows = "N/A"

                datasets.append({
                    "filename": fname,
                    "size_mb": size_mb,
                    "total_rows": total_rows,
                    "columns": columns,
                    "preview": preview,
                    "class_distribution": class_counts
                })
            except Exception as e:
                datasets.append({
                    "filename": fname,
                    "size_mb": size_mb,
                    "error": str(e)
                })
        self.send_json({"datasets": datasets})

    def handle_api_simulations(self):
        results = []
        files = glob.glob(os.path.join(RESULTS_DIR, "simulation_results_*.json"))
        # Sort newest first
        files.sort(key=os.path.getmtime, reverse=True)

        for filepath in files:
            fname = os.path.basename(filepath)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                experiments = data.get("experiments", {})
                
                summary_exps = {}
                for exp_name, exp_data in experiments.items():
                    accs = exp_data.get("accuracy", [])
                    losses = exp_data.get("loss", [])
                    rmses = exp_data.get("rmse", [])
                    aucs = exp_data.get("auc", [])
                    f1s = exp_data.get("f1", [])
                    summary_exps[exp_name] = {
                        "final_acc": accs[-1] if accs else None,
                        "final_loss": losses[-1] if losses else None,
                        "final_rmse": rmses[-1] if rmses else None,
                        "final_auc": aucs[-1] if aucs else None,
                        "final_f1": f1s[-1] if f1s else None,
                        "rounds_completed": len(losses)
                    }

                # Find corresponding report and image files
                content_name = fname.replace("simulation_results_", "").replace(".json", "")
                report_file = os.path.join(REPORTS_DIR, f"simulation_report_{content_name}.md")
                model_plot = os.path.join(REPORTS_DIR, f"model_comparison_{content_name}.png")
                strat_plot = os.path.join(REPORTS_DIR, f"strategy_comparison_{content_name}.png")

                results.append({
                    "filename": fname,
                    "content_name": content_name,
                    "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(filepath))),
                    "metadata": meta,
                    "experiments": summary_exps,
                    "has_report": os.path.exists(report_file),
                    "report_file": f"/reports/simulation_report_{content_name}.md" if os.path.exists(report_file) else None,
                    "model_plot": f"/reports/model_comparison_{content_name}.png" if os.path.exists(model_plot) else None,
                    "strat_plot": f"/reports/strategy_comparison_{content_name}.png" if os.path.exists(strat_plot) else None
                })
            except Exception as e:
                results.append({"filename": fname, "error": str(e)})

        self.send_json({"simulations": results})

    def handle_api_simulation_detail(self, query):
        fname = query.get("file", [""])[0]
        if not fname:
            self.send_json({"error": "Missing 'file' parameter"}, 400)
            return

        filepath = os.path.join(RESULTS_DIR, os.path.basename(fname))
        if not os.path.exists(filepath):
            self.send_json({"error": "File not found"}, 404)
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_api_reports(self):
        reports = []
        for file in glob.glob(os.path.join(REPORTS_DIR, "simulation_report_*.md")):
            fname = os.path.basename(file)
            cname = fname.replace("simulation_report_", "").replace(".md", "")
            reports.append({
                "filename": fname,
                "content_name": cname,
                "url": f"/reports/{fname}",
                "model_plot": f"/reports/model_comparison_{cname}.png" if os.path.exists(os.path.join(REPORTS_DIR, f"model_comparison_{cname}.png")) else None,
                "strategy_plot": f"/reports/strategy_comparison_{cname}.png" if os.path.exists(os.path.join(REPORTS_DIR, f"strategy_comparison_{cname}.png")) else None,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file)))
            })
        reports.sort(key=lambda x: x["modified"], reverse=True)
        self.send_json({"reports": reports})

    def handle_api_report_view(self, query):
        fname = query.get("file", [""])[0]
        if not fname:
            self.send_json({"error": "Missing file"}, 400)
            return
        filepath = os.path.join(REPORTS_DIR, os.path.basename(fname))
        if not os.path.exists(filepath):
            self.send_json({"error": "File not found"}, 404)
            return
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        self.send_json({"content": content})

    def handle_api_simulation_start(self):
        global simulation_thread
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            params = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            params = {}

        with simulation_lock:
            if sim_state["is_running"]:
                self.send_json({"error": "A simulation is currently in progress. Please wait or stop it first."}, 409)
                return

            simulation_thread = threading.Thread(target=simulation_worker, args=(params,), daemon=True)
            simulation_thread.start()

        self.send_json({"status": "started", "message": "Simulation started successfully."})

    def handle_api_simulation_stop(self):
        if not sim_state["is_running"]:
            self.send_json({"message": "No simulation is currently running."})
            return
        simulation_stop_event.set()
        log_message("User requested simulation stop.")
        self.send_json({"message": "Stop signal sent to simulation."})


def run_server(port=PORT):
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    current_port = port
    while current_port < port + 20:
        try:
            server = ThreadingHTTPServer((HOST, current_port), FLServerHandler)
            print("==================================================")
            print(f"[+] HealthCare Federated Learning Studio is LIVE!")
            print(f"[+] Access Web Dashboard: http://localhost:{current_port}")
            print("==================================================")
            server.serve_forever()
            break
        except OSError:
            print(f"Port {current_port} in use, trying {current_port + 1}...")
            current_port += 1

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(p)

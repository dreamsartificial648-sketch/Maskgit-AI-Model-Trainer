from __future__ import annotations

import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageDraw, ImageTk

from maskgit.data import find_images, tensor_to_pil
from maskgit.engine import build_token_cache, generate_images, train_full_pipeline, train_maskgit, train_vqvae


ROOT = Path(__file__).resolve().parent

RESOLUTION_PRESETS = {
    128: (8, 16, 24, 384, 8, 8),
    256: (16, 8, 12, 384, 8, 8),
    384: (16, 4, 4, 288, 6, 6),
    512: (16, 2, 2, 256, 6, 8),
    768: (16, 1, 1, 192, 4, 6),
}


class MaskGITApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MaskGIT Lab")
        self.geometry("1120x760")
        self.minsize(940, 650)
        self.configure(bg="#11151c")
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.preview_photo = None
        self.vars = {}
        self._style()
        self._build()
        self.after(100, self._poll)

    def _style(self):
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        bg, panel, fg, muted, accent = "#11151c", "#1a202b", "#edf2f7", "#9aa7b8", "#7c9cff"
        style.configure(".", background=bg, foreground=fg, fieldbackground="#242c39", font=("Segoe UI", 10))
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=panel)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Card.TLabel", background=panel, foreground=fg)
        style.configure("Muted.TLabel", background=panel, foreground=muted)
        style.configure("Title.TLabel", background=bg, foreground=fg, font=("Segoe UI Semibold", 24))
        style.configure("TButton", padding=(12, 7), background="#2b3545", foreground=fg)
        style.map("TButton", background=[("active", "#39465a")])
        style.configure("Accent.TButton", background=accent, foreground="#081020", font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#9ab0ff")])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), background="#202735", foreground=muted)
        style.map("TNotebook.Tab", background=[("selected", panel)], foreground=[("selected", fg)])
        style.configure("Horizontal.TProgressbar", troughcolor="#242c39", background=accent)

    def _build(self):
        top = ttk.Frame(self, padding=(24, 18, 24, 8)); top.pack(fill="x")
        ttk.Label(top, text="MaskGIT Lab", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="Train a fast iterative image generator from your own image folder.", foreground="#9aa7b8").pack(side="left", padx=18, pady=(8, 0))
        ttk.Button(top, text="Load project", command=self.load_project).pack(side="right")
        ttk.Button(top, text="Save project", command=self.save_project).pack(side="right", padx=8)
        body = ttk.Frame(self, padding=(24, 8, 24, 20)); body.pack(fill="both", expand=True)
        left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, style="Card.TFrame", padding=18); right.pack(side="right", fill="both", padx=(18, 0)); right.configure(width=360); right.pack_propagate(False)
        self.notebook = ttk.Notebook(left); self.notebook.pack(fill="both", expand=True)
        self._tokenizer_tab(); self._transformer_tab(); self._generate_tab()
        self._status_panel(right)

    def _card(self, parent, title, subtitle=""):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18); card.pack(fill="x", padx=6, pady=7)
        ttk.Label(card, text=title, style="Card.TLabel", font=("Segoe UI Semibold", 13)).grid(row=0, column=0, columnspan=4, sticky="w")
        if subtitle: ttk.Label(card, text=subtitle, style="Muted.TLabel", wraplength=590).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 13))
        card.columnconfigure(1, weight=1); card.columnconfigure(3, weight=1)
        return card, 2

    def _field(self, parent, row, label, key, default, kind="entry", column=0):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=column, sticky="w", padx=(0, 8), pady=5)
        variable = tk.StringVar(value=str(default)); self.vars[key] = variable
        if kind == "check":
            variable = tk.BooleanVar(value=bool(default)); self.vars[key] = variable
            widget = ttk.Checkbutton(parent, variable=variable)
        else: widget = ttk.Entry(parent, textvariable=variable)
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=5)
        return widget

    def _path_field(self, parent, row, label, key, default, folder=False, save=False):
        self._field(parent, row, label, key, default)
        def browse():
            if folder: result = filedialog.askdirectory(initialdir=self.vars[key].get() or ROOT)
            elif save: result = filedialog.asksaveasfilename(initialdir=ROOT, defaultextension=".pt", filetypes=[("PyTorch files", "*.pt")])
            else: result = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")])
            if result: self.vars[key].set(result)
        ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, sticky="w", pady=5)

    def _tokenizer_tab(self):
        tab = ttk.Frame(self.notebook, padding=8); self.notebook.add(tab, text="1  Tokenizer")
        card, row = self._card(tab, "Dataset and output", "The tokenizer learns a compact vocabulary of visual patches. Images are center-cropped automatically.")
        self._path_field(card, row, "Image folder", "dataset", "", folder=True); row += 1
        self._path_field(card, row, "Project output", "output", str(ROOT / "runs"), folder=True)
        card, row = self._card(tab, "Tokenizer settings", "Defaults target 128×128 images and fit comfortably on a 12 GB GPU.")
        self._field(card, row, "Image size", "image_size", 128); self._field(card, row, "Batch size", "vq_batch", 16, column=2); row += 1
        self._field(card, row, "Epochs", "vq_epochs", 50); self._field(card, row, "Learning rate", "vq_lr", 0.0002, column=2); row += 1
        self._field(card, row, "Codebook entries", "codebook", 512); self._field(card, row, "Hidden channels", "hidden", 128, column=2); row += 1
        self._field(card, row, "Embedding size", "embed", 128); self._field(card, row, "Use NVIDIA GPU", "cuda", True, kind="check", column=2); row += 1
        self._field(card, row, "VQGAN sharpening", "vqgan", True, kind="check"); self._field(card, row, "GAN strength", "adv_weight", 0.1, column=2)
        buttons = ttk.Frame(tab); buttons.pack(fill="x", padx=6, pady=10)
        ttk.Button(buttons, text="Train full pipeline", style="Accent.TButton", command=self.start_full_pipeline).pack(side="left")
        ttk.Button(buttons, text="Tokenizer only", command=self.start_vq).pack(side="left", padx=8)
        ttk.Button(buttons, text="Build token cache", command=self.start_cache).pack(side="left", padx=8)
        ttk.Button(buttons, text="Optimize for resolution", command=self.optimize_resolution).pack(side="right")

    def _transformer_tab(self):
        tab = ttk.Frame(self.notebook, padding=8); self.notebook.add(tab, text="2  MaskGIT")
        card, row = self._card(tab, "Token data", "Build the cache after tokenizer training, then train the bidirectional transformer.")
        self._path_field(card, row, "Token cache", "cache", str(ROOT / "runs" / "tokens.pt")); row += 1
        card2, row = self._card(tab, "Transformer settings", "A mask is sampled anew for every image on every training step.")
        self._field(card2, row, "Batch size", "mg_batch", 24); self._field(card2, row, "Epochs", "mg_epochs", 100, column=2); row += 1
        self._field(card2, row, "Model width", "dimension", 384); self._field(card2, row, "Layers", "layers", 8, column=2); row += 1
        self._field(card2, row, "Attention heads", "heads", 8); self._field(card2, row, "Dropout", "dropout", 0.1, column=2); row += 1
        self._field(card2, row, "Learning rate", "mg_lr", 0.0002)
        buttons = ttk.Frame(tab); buttons.pack(fill="x", padx=6, pady=10)
        ttk.Button(buttons, text="Train MaskGIT", style="Accent.TButton", command=self.start_maskgit).pack(side="left")

    def _generate_tab(self):
        tab = ttk.Frame(self.notebook, padding=8); self.notebook.add(tab, text="3  Generate")
        card, row = self._card(tab, "Checkpoints", "Load the two trained networks. Generation begins fully masked and reveals the most confident tokens over several passes.")
        self._path_field(card, row, "Tokenizer", "vq_path", str(ROOT / "runs" / "vqvae_final.pt")); row += 1
        self._path_field(card, row, "MaskGIT", "mg_path", str(ROOT / "runs" / "maskgit_final.pt")); row += 1
        self._path_field(card, row, "Save images to", "gen_output", str(ROOT / "generated"), folder=True)
        card, row = self._card(tab, "Generation settings")
        self._field(card, row, "Image count", "gen_count", 4); self._field(card, row, "Refinement passes", "gen_steps", 12, column=2); row += 1
        self._field(card, row, "Temperature", "temperature", 1.0)
        buttons = ttk.Frame(tab); buttons.pack(fill="x", padx=6, pady=10)
        ttk.Button(buttons, text="Generate images", style="Accent.TButton", command=self.start_generate).pack(side="left")

    def _status_panel(self, parent):
        ttk.Label(parent, text="Preview", style="Card.TLabel", font=("Segoe UI Semibold", 14)).pack(anchor="w")
        self.preview = tk.Label(parent, text="Training reconstructions and generated\nimages will appear here.", bg="#10141b", fg="#738096", font=("Segoe UI", 10))
        self.preview.pack(fill="x", pady=(10, 16)); self.preview.configure(height=18)
        self.progress = ttk.Progressbar(parent, maximum=1.0); self.progress.pack(fill="x")
        self.status = ttk.Label(parent, text="Ready", style="Muted.TLabel", wraplength=320); self.status.pack(anchor="w", pady=(7, 12))
        ttk.Label(parent, text="Activity", style="Card.TLabel", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.log = tk.Text(parent, height=10, bg="#10141b", fg="#b8c2d1", insertbackground="white", relief="flat", wrap="word", padx=10, pady=10, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(8, 10)); self.log.configure(state="disabled")
        self.stop_button = ttk.Button(parent, text="Stop safely", command=self.stop, state="disabled"); self.stop_button.pack(anchor="e")

    def _int(self, key): return int(self.vars[key].get())
    def _float(self, key): return float(self.vars[key].get())

    def _common(self):
        return {"output": self.vars["output"].get(), "cuda": self.vars["cuda"].get()}

    def _training_config(self):
        size = self._int("image_size")
        downsample = RESOLUTION_PRESETS.get(size, (8 if size <= 128 else 16,))[0]
        if size % downsample:
            raise ValueError(f"Image size must be divisible by {downsample}.")
        return {**self._common(), "dataset": self.vars["dataset"].get(), "image_size": size,
                "codebook_size": self._int("codebook"), "hidden": self._int("hidden"),
                "embedding_dim": self._int("embed"), "downsample_factor": downsample,
                "vqgan": self.vars["vqgan"].get(), "adversarial_weight": self._float("adv_weight")}

    def start_vq(self):
        try: cfg = self._training_config()
        except ValueError as error: messagebox.showerror("Check settings", str(error)); return
        cfg.update(batch_size=self._int("vq_batch"), epochs=self._int("vq_epochs"), learning_rate=self._float("vq_lr"))
        self._start(train_vqvae, cfg, "Tokenizer training")

    def start_full_pipeline(self):
        try:
            cfg = self._training_config()
            cfg.update(cache=self.vars["cache"].get(), vq_batch_size=self._int("vq_batch"),
                       vq_epochs=self._int("vq_epochs"), vq_learning_rate=self._float("vq_lr"),
                       mg_batch_size=self._int("mg_batch"), mg_epochs=self._int("mg_epochs"),
                       mg_learning_rate=self._float("mg_lr"), dimension=self._int("dimension"),
                       layers=self._int("layers"), heads=self._int("heads"), dropout=self._float("dropout"))
        except ValueError as error: messagebox.showerror("Check settings", str(error)); return
        self._start(train_full_pipeline, cfg, "Full training pipeline")

    def start_cache(self):
        output = Path(self.vars["output"].get()); final = output / "vqvae_final.pt"; latest = output / "vqvae_latest.pt"
        cfg = {**self._common(), "dataset": self.vars["dataset"].get(), "vqvae": str(final if final.exists() else latest),
               "cache": self.vars["cache"].get(), "batch_size": self._int("vq_batch"), "image_size": self._int("image_size")}
        self._start(build_token_cache, cfg, "Token encoding")

    def start_maskgit(self):
        cfg = {**self._common(), "cache": self.vars["cache"].get(), "batch_size": self._int("mg_batch"),
               "epochs": self._int("mg_epochs"), "learning_rate": self._float("mg_lr"), "dimension": self._int("dimension"),
               "layers": self._int("layers"), "heads": self._int("heads"), "dropout": self._float("dropout")}
        self._start(train_maskgit, cfg, "MaskGIT training")

    def start_generate(self):
        cfg = {"vqvae": self.vars["vq_path"].get(), "maskgit": self.vars["mg_path"].get(), "output": self.vars["gen_output"].get(),
               "count": self._int("gen_count"), "steps": self._int("gen_steps"), "temperature": self._float("temperature"), "cuda": self.vars["cuda"].get()}
        self._start(generate_images, cfg, "Generation")

    def optimize_resolution(self):
        try: size = self._int("image_size")
        except ValueError: messagebox.showerror("Check settings", "Image size must be a whole number."); return
        preset_size = min(RESOLUTION_PRESETS, key=lambda candidate: abs(candidate - size))
        downsample, vq_batch, mg_batch, dimension, layers, heads = RESOLUTION_PRESETS[preset_size]
        for key, value in {"vq_batch": vq_batch, "mg_batch": mg_batch, "dimension": dimension,
                           "layers": layers, "heads": heads}.items(): self.vars[key].set(str(value))
        messagebox.showinfo("Settings optimized", f"Applied the {preset_size}px performance preset.\n"
                            f"The tokenizer will use {downsample}× compression to keep token grids manageable.")

    def save_project(self):
        target = filedialog.asksaveasfilename(initialdir=self.vars["output"].get() or str(ROOT),
            initialfile="maskgit_project.json", defaultextension=".json", filetypes=[("MaskGIT project", "*.json")])
        if not target: return
        payload = {"format": "MaskGIT Lab project", "version": 1,
                   "settings": {key: value.get() for key, value in self.vars.items()}}
        Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append(f"Project settings saved: {target}")

    def load_project(self):
        source = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("MaskGIT project", "*.json")])
        if not source: return
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            if payload.get("format") != "MaskGIT Lab project": raise ValueError("This is not a MaskGIT Lab project file.")
            for key, value in payload["settings"].items():
                if key in self.vars: self.vars[key].set(value)
            self._append(f"Project settings loaded: {source}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            messagebox.showerror("Could not load project", str(error))

    def _start(self, function, config, label):
        if self.worker and self.worker.is_alive(): messagebox.showinfo("Already running", "Stop or finish the current job first."); return
        try:
            if "dataset" in config and not Path(config["dataset"]).is_dir(): raise ValueError("Choose a valid image folder.")
            # A cache is an input for MaskGIT training, but it is the output of
            # the "Build token cache" action and therefore must not exist yet.
            required_files = ("vqvae", "maskgit")
            if function is not build_token_cache and function is not train_full_pipeline:
                required_files += ("cache",)
            for key in required_files:
                if key in config and not Path(config[key]).is_file(): raise ValueError(f"Choose a valid {key} file.")
            if config.get("dimension") and config["dimension"] % config["heads"]: raise ValueError("Model width must be divisible by attention heads.")
        except (ValueError, OSError) as error: messagebox.showerror("Check settings", str(error)); return
        self.stop_event.clear(); self.progress["value"] = 0; self.stop_button["state"] = "normal"; self.status["text"] = f"{label} started…"
        self._append(f"{label} started.")
        def run():
            try:
                result = function(config, lambda kind, data: self.events.put((kind, data)), self.stop_event)
                self.events.put(("done", {"result": result, "label": label}))
            except Exception as error:
                self.events.put(("error", {"text": str(error), "trace": traceback.format_exc()}))
        self.worker = threading.Thread(target=run, daemon=True); self.worker.start()

    def stop(self):
        self.stop_event.set(); self.status["text"] = "Stopping after the current batch…"; self.stop_button["state"] = "disabled"

    def _append(self, text):
        self.log.configure(state="normal"); self.log.insert("end", text + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "log": self._append(data["text"])
                elif kind == "progress": self.progress["value"] = data["value"]; self.status["text"] = data["text"]
                elif kind == "preview_tensor": self._show_grid(data["tensor"])
                elif kind == "done":
                    self.stop_button["state"] = "disabled"; self.progress["value"] = 1
                    self.status["text"] = f"{data['label']} finished"; self._append(f"Finished: {data['result']}")
                    if data["label"] == "Tokenizer training" and data["result"] != "stopped": self.vars["vq_path"].set(data["result"])
                    elif data["label"] == "Token encoding" and data["result"] != "stopped": self.vars["cache"].set(data["result"])
                    elif data["label"] == "MaskGIT training" and data["result"] != "stopped": self.vars["mg_path"].set(data["result"])
                    elif data["label"] == "Full training pipeline" and data["result"].get("status") == "complete":
                        result = data["result"]
                        self.vars["vq_path"].set(result["vqvae"])
                        self.vars["cache"].set(result["cache"])
                        self.vars["mg_path"].set(result["maskgit"])
                        self._append("All model files are ready. Save a project file to load this setup later.")
                elif kind == "error":
                    self.stop_button["state"] = "disabled"; self.status["text"] = "An error occurred"; self._append(data["trace"])
                    messagebox.showerror("Job failed", data["text"])
        except queue.Empty: pass
        self.after(100, self._poll)

    def _show_grid(self, tensors: torch.Tensor):
        images = [tensor_to_pil(t) for t in tensors[:8]]
        columns = 2 if len(images) <= 4 else 4; cell = 150; rows = (len(images) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell, rows * cell), "#10141b")
        for index, image in enumerate(images):
            image.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
            x = (index % columns) * cell + (cell - image.width) // 2; y = (index // columns) * cell + (cell - image.height) // 2
            canvas.paste(image, (x, y))
        canvas.thumbnail((324, 320), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(canvas); self.preview.configure(image=self.preview_photo, text="", height=canvas.height)


if __name__ == "__main__":
    MaskGITApp().mainloop()

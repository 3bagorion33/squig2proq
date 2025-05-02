import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
import importlib.resources as pkg_resources
import importlib.resources
from squig2proq import data
from squig2proq.parser import parse_filter_text, build_ffp, truncate_middle
import os
import ctypes.wintypes
import json
import sys

CONFIG_DIR = Path.home() / ".squig2proq"
CONFIG_PATH = CONFIG_DIR / "config.json"

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Error reading config:", e)
    return {"last_file": "", "export_dir": "", "last_file_name": ""}

def save_config(file_path, export_dir, file_name):
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump({"last_file": file_path, "export_dir": export_dir, "last_file_name": file_name}, f)
    except Exception as e:
        print("Error saving config:", e)

def get_default_install_path():
    SHGFP_TYPE_CURRENT = 0
    CSIDL_PERSONAL = 5  # "My Documents"
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    if ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf) != 0:
        raise OSError("Cannot find Documents folder")
    documents = Path(buf.value) / "FabFilter" / "Presets" / "Pro-Q 4"
    documents.mkdir(parents=True, exist_ok=True)
    return str(documents)

def launch_gui():
    root = TkinterDnD.Tk()
    root.title("Squig2ProQ")

    status_var = tk.StringVar(value="Ready")
    
    def update_status(message):
        status_var.set(message)

    status_label = ttk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def print_status(message):
        print(message)
        update_status(message)

    # Replace all print calls with print_status
    def load_from_path(path, update_file_name=True):
        nonlocal filters
        current_file.set(path)
        # Removed functionality to update file_name when loading a new file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            filters = parse_filter_text(content)
            for row in tree.get_children():
                tree.delete(row)
            for f in filters:
                tree.insert('', tk.END, values=(f['Frequency'], f['Gain'], f['Q']))
            print_status(f"Loaded file: {path}")
        except Exception as e:
            print_status(f"Error loading file: {e}")

    def load_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")],
            initialdir=Path(current_file.get()).parent if current_file.get() else None
        )
        if file_path:
            load_from_path(file_path, update_file_name=True)

    def choose_file():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".ffp",
            filetypes=[("FabFilter Preset", "*.ffp"), ("All Files", "*.*")],
            initialdir=save_dir.get(),
            initialfile=file_name.get()
        )
        if file_path:
            save_dir.set(str(Path(file_path).parent))
            file_name.set(Path(file_path).stem)
            path_label.config(text=truncate_middle(file_path))

    def export_ffp():
        if not filters:
            print_status("No filters to export.")
            return
        try:
            if getattr(sys, 'frozen', False):
                # If running in a PyInstaller bundle
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            template_path = os.path.join(base_path, "squig2proq", "data", "Clean.ffp")
            if not os.path.exists(template_path):
                # Adjust path for PyInstaller's handling of --add-data
                template_path = os.path.join(base_path, "data", "Clean.ffp")
            with open(template_path, 'r', encoding='utf-8') as template_file:
                template_text = template_file.read()
        except Exception as e:
            print_status(f"Template read error: {e}")
            return
        result = build_ffp(filters, template_text)

        filename = file_name.get().strip()
        directory = save_dir.get().strip()
        if not filename:
            print_status("Filename is empty")
            return
        if not directory:
            print_status("Directory not selected")
            return

        output_path = Path(directory) / f"{filename}.ffp"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(result)
            print_status(f"Saved to {output_path}")
            save_config(current_file.get(), save_dir.get(), file_name.get())
        except Exception as e:
            print_status(f"Error saving file: {e}")

    filters = []
    config = load_config()

    def on_drop(event):
        try:
            path = root.tk.splitlist(event.data)[0]
            if path.lower().endswith(".txt"):
                load_from_path(path, update_file_name=True)
        except Exception as e:
            print_status(f"Drag-and-drop error: {e}")

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', on_drop)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    btn = ttk.Button(frame, text="Load Filter File", command=load_file)
    btn.pack(pady=5)

    current_file = tk.StringVar(value=config.get("last_file", ""))

    export_frame = ttk.Frame(frame)
    export_frame.pack(pady=5, fill=tk.X)

    file_name = tk.StringVar(value=config.get("last_file_name", "Parsed_Filter"))
    default_path = config.get("export_dir") or get_default_install_path()
    save_dir = tk.StringVar(value=default_path)

    name_entry = ttk.Entry(export_frame, textvariable=file_name, width=30)
    name_entry.pack(side=tk.LEFT, padx=(0, 5))

    path_label = ttk.Label(export_frame, text=truncate_middle(default_path), anchor="w")
    path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    def update_path_label():
        label_width = path_label.winfo_width()
        truncated_text = truncate_middle(save_dir.get(), label_width // 7)  # Approximate character width
        path_label.config(text=truncated_text)

    path_label.bind('<Configure>', lambda e: update_path_label())

    export_btn = ttk.Button(export_frame, text="Export", command=export_ffp)
    export_btn.pack(side=tk.RIGHT)

    file_btn = ttk.Button(export_frame, text="Save to File", command=choose_file)
    file_btn.pack(side=tk.RIGHT, padx=(5, 0))

    columns = ('Frequency', 'Gain', 'Q')
    global tree
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    tree.pack(fill=tk.BOTH, expand=True)

    if current_file.get():
        try:
            load_from_path(current_file.get(), update_file_name=False)
        except Exception as e:
            print_status(f"Failed to load last file: {e}")

    root.mainloop()

def main():
    launch_gui()

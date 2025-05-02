import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
from squig2proq.parser import truncate_middle
from squig2proq.main import (
    load_config, save_config, get_default_install_path,
    load_from_path, load_file, choose_file, export_ffp, on_drop, save_ir_wav, export_ir
)

def launch_gui():
    root = TkinterDnD.Tk()
    root.title("Squig2ProQ")

    # --- Переменные состояния ---
    config = load_config()
    status_var = tk.StringVar(value="Ready")
    current_file = tk.StringVar(value=config.get("last_file", ""))
    file_name = tk.StringVar(value=config.get("last_file_name", "Parsed_Filter"))
    default_path = config.get("export_dir") or get_default_install_path()
    save_dir = tk.StringVar(value=default_path)
    adjust_q = tk.BooleanVar(value=config.get("adjust_q", False))
    ir_type_options = ["Linear Phase", "Minimum Phase", "Mixed Phase"]
    ir_type = tk.StringVar(value=config.get("ir_type", ir_type_options[0]))
    fs_options = [44100, 48000, 88200, 96000, 176400, 192000]
    ir_fs = tk.IntVar(value=int(config.get("ir_fs", 48000)))
    preamp = tk.DoubleVar(value=config.get("preamp", 0.0))
    tilt = tk.DoubleVar(value=config.get("tilt", 0.0))
    subsonic = tk.BooleanVar(value=config.get("subsonic", False))
    ir_save_dir = tk.StringVar(value=config.get("ir_export_dir", save_dir.get()))

    # --- Виджеты ---
    status_label = ttk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    top_row = ttk.Frame(frame)
    top_row.pack(fill=tk.X)
    btn = ttk.Button(top_row, text="Load AutoEQ File", command=lambda: load_file(state))
    btn.pack(side=tk.LEFT)
    adjust_q_checkbox = ttk.Checkbutton(top_row, text="Correct Q for FabFilter", variable=adjust_q)
    adjust_q_checkbox.pack(side=tk.RIGHT, padx=(10, 0))

    export_frame = ttk.Frame(frame)
    export_frame.pack(pady=5, fill=tk.X)
    name_entry = ttk.Entry(export_frame, textvariable=file_name, width=30)
    name_entry.pack(side=tk.LEFT, padx=(0, 5))
    path_label = ttk.Label(export_frame, text=truncate_middle(default_path), anchor="w")
    path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    export_btn = ttk.Button(export_frame, text="Export", command=lambda: export_ffp(state))
    export_btn.pack(side=tk.RIGHT)
    file_btn = ttk.Button(export_frame, text="Save to .ffp", command=lambda: choose_file(state))
    file_btn.pack(side=tk.RIGHT, padx=(5, 0))

    ir_frame = ttk.Frame(frame)
    ir_frame.pack(pady=5, fill=tk.X)
    ir_type_combo = ttk.Combobox(ir_frame, textvariable=ir_type, values=ir_type_options, width=16, state="readonly")
    ir_type_combo.pack(side=tk.LEFT, padx=(0, 5))
    fs_combo = ttk.Combobox(ir_frame, textvariable=ir_fs, values=fs_options, width=6, state="readonly")
    fs_combo.pack(side=tk.LEFT, padx=(0, 5))
    ir_path_label = ttk.Label(ir_frame, text=truncate_middle(ir_save_dir.get()), anchor="w")
    ir_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    ir_export_btn = ttk.Button(ir_frame, text="Export", command=lambda: export_ir(state))
    ir_export_btn.pack(side=tk.RIGHT)
    ir_save_btn = ttk.Button(ir_frame, text="Save to .wav", command=lambda: save_ir_wav(state))
    ir_save_btn.pack(side=tk.RIGHT, padx=(5, 0))

    param_frame = ttk.Frame(frame)
    param_frame.pack(pady=2, fill=tk.X)
    preamp_label = ttk.Label(param_frame, text="Preamp (dB):")
    preamp_label.pack(side=tk.LEFT, padx=(0, 0))
    preamp_spinbox = tk.Spinbox(param_frame, textvariable=preamp, from_=-24.0, to=24.0, increment=0.1, width=6, format="%.1f")
    preamp_spinbox.pack(side=tk.LEFT, padx=(0, 5))
    tilt_label = ttk.Label(param_frame, text="Tilt (dB):")
    tilt_label.pack(side=tk.LEFT, padx=(0, 0))
    tilt_spinbox = tk.Spinbox(param_frame, textvariable=tilt, from_=-12.0, to=12.0, increment=0.1, width=6, format="%.1f")
    tilt_spinbox.pack(side=tk.LEFT, padx=(0, 5))
    subsonic_checkbox = ttk.Checkbutton(param_frame, text="SubSonic", variable=subsonic)
    subsonic_checkbox.pack(side=tk.RIGHT, padx=(10, 0))

    columns = ('Type', 'Frequency', 'Gain', 'Q')
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    tree.pack(fill=tk.BOTH, expand=True)

    # --- Словарь состояния для передачи в main.py ---
    state = dict(
        root=root,
        status_var=status_var,
        current_file=current_file,
        file_name=file_name,
        save_dir=save_dir,
        adjust_q=adjust_q,
        ir_type=ir_type,
        ir_fs=ir_fs,
        preamp=preamp,
        tilt=tilt,
        subsonic=subsonic,
        ir_save_dir=ir_save_dir,
        tree=tree,
        path_label=path_label,
        ir_path_label=ir_path_label,
    )

    # --- Привязки событий ---
    path_label.bind('<Configure>', lambda e: path_label.config(text=truncate_middle(save_dir.get(), path_label.winfo_width() // 7)))
    ir_path_label.bind('<Configure>', lambda e: ir_path_label.config(text=truncate_middle(ir_save_dir.get(), ir_path_label.winfo_width() // 7)))
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', lambda event: on_drop(event, state))

    # --- Запуск ---
    if current_file.get():
        try:
            load_from_path(current_file.get(), False, state)
        except Exception:
            pass
    root.mainloop()

def main():
    launch_gui()

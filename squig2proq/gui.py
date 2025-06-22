import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
from squig2proq.parser import truncate_middle
from squig2proq.main import (
    load_config, save_config, get_default_install_path,
    load_from_path, load_file, choose_file, export_ffp, on_drop, save_ir_wav, save_link_wav, export_ir
)

def launch_gui():
    root = TkinterDnD.Tk()
    root.title("Squig2ProQ")

    # --- Координаты и размеры окна из конфига (глобальные) ---
    config = load_config()
    if all(config.get(k) is not None for k in ("window_x", "window_y", "window_width", "window_height")):
        try:
            geom = f"{int(config['window_width'])}x{int(config['window_height'])}+{int(config['window_x'])}+{int(config['window_y'])}"
            root.geometry(geom)
        except Exception:
            pass

    # --- Переменные состояния ---
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
    subsonic = tk.BooleanVar(value=config.get("subsonic", True))    
    subsonic_freq = tk.DoubleVar(value=config.get("subsonic_freq", 10.0))
    ir_save_dir = tk.StringVar(value=config.get("ir_export_dir", save_dir.get()))
    link_ir = tk.BooleanVar(value=config.get("link_ir", False))
    link_wav_dir = tk.StringVar(value=config.get("link_wav_dir", save_dir.get()))

    # --- Восстановление множественного выбора fs и ir_type ---
    fs_selected = config.get("fs_selected", [ir_fs.get()])
    ir_type_selected = config.get("ir_type_selected", [ir_type.get()])

    # --- Кастомный выпадающий список с чекбоксами ---
    def create_multicheck_button(parent, options, var_list, label_text, width=16):
        btn_var = tk.StringVar()
        btn = ttk.Button(parent, text=label_text, width=width)
        def update_btn_text():
            selected = [opt for opt, v in zip(options, var_list) if v.get()]
            btn_var.set(', '.join(map(str, selected)) if selected else label_text)
            btn.config(text=btn_var.get())
        def show_menu():
            # Получаем координаты кнопки
            bx = btn.winfo_rootx()
            by = btn.winfo_rooty()
            bh = btn.winfo_height()
            # Создаём окно и позиционируем под кнопкой
            top = tk.Toplevel(parent)
            top.transient(parent)
            top.grab_set()
            top.title(label_text)
            # Позиционирование: левый нижний угол кнопки
            top.update_idletasks()
            top.geometry(f'+{bx}+{by+bh}')
            for i, opt in enumerate(options):
                cb = ttk.Checkbutton(top, text=str(opt), variable=var_list[i], command=update_btn_text)
                cb.pack(anchor='w', padx=10, pady=2)
            def on_close():
                update_btn_text()
                top.destroy()
            top.protocol('WM_DELETE_WINDOW', on_close)
        btn.config(command=show_menu)
        update_btn_text()
        return btn

    # --- Виджеты ---
    status_label = ttk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    load_frame = ttk.Frame(frame)
    load_frame.pack(fill=tk.X)

    name_entry = ttk.Entry(load_frame, textvariable=file_name, width=93)
    name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    load_btn = ttk.Button(load_frame, text="Load AutoEQ File", command=lambda: load_file(state))
    load_btn.pack(side=tk.RIGHT, padx=(0, 0))

    override_name = tk.BooleanVar(value=config.get("override_name", False))
    override_checkbox = ttk.Checkbutton(load_frame, text="Override name", variable=override_name)
    override_checkbox.pack(side=tk.RIGHT, padx=(0, 5))
 
    ffp_frame = ttk.Frame(frame)
    ffp_frame.pack(pady=5, fill=tk.X)

    path_label = ttk.Label(ffp_frame, text=truncate_middle(default_path), anchor="w")
    path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    ffp_export_btn = ttk.Button(ffp_frame, text="Export", command=lambda: export_ffp(state))
    ffp_export_btn.pack(side=tk.RIGHT)

    ffp_dir_btn = ttk.Button(ffp_frame, text="Dir for .ffp", command=lambda: choose_file(state))
    ffp_dir_btn.pack(side=tk.RIGHT, padx=(5, 0))
    adjust_q_checkbox = ttk.Checkbutton(ffp_frame, text="Correct Q for FabFilter", variable=adjust_q)
    adjust_q_checkbox.pack(side=tk.RIGHT, padx=(0, 5))

    ir_frame = ttk.Frame(frame)
    ir_frame.pack(pady=5, fill=tk.X)

    ir_path_label = ttk.Label(ir_frame, text=truncate_middle(ir_save_dir.get()), anchor="w")
    ir_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    ir_export_btn = ttk.Button(ir_frame, text="Export", command=lambda: export_ir(state))
    ir_export_btn.pack(side=tk.RIGHT)
    ir_dir_btn = ttk.Button(ir_frame, text="Dir for .wav", command=lambda: save_ir_wav(state))
    ir_dir_btn.pack(side=tk.RIGHT, padx=(5, 0))

    # --- Множественный выбор для fs_options ---
    fs_var_list = [tk.BooleanVar(value=(fs in fs_selected)) for fs in fs_options]
    fs_btn = create_multicheck_button(ir_frame, fs_options, fs_var_list, "Sample Rate", width=12)
    fs_btn.pack(side=tk.RIGHT, padx=(0, 5))

    # --- Множественный выбор для ir_type_options ---
    ir_type_var_list = [tk.BooleanVar(value=(t in ir_type_selected)) for t in ir_type_options]
    ir_type_btn = create_multicheck_button(ir_frame, ir_type_options, ir_type_var_list, "IR Type", width=16)
    ir_type_btn.pack(side=tk.RIGHT, padx=(0, 5))
       
    link_frame = ttk.Frame(frame)
    link_frame.pack(pady=5, fill=tk.X)
    
    link_wav_dir = tk.StringVar(value=config.get("link_wav_dir", save_dir.get()))
    link_path_label = ttk.Label(link_frame, text=truncate_middle(link_wav_dir.get()), anchor="w")
    link_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    link_ir_checkbox = ttk.Checkbutton(link_frame, text="Link for IR", variable=link_ir)
    link_ir_checkbox.pack(side=tk.RIGHT, padx=(0, 0))

    link_dir_btn = ttk.Button(link_frame, text="Dir for link", command=lambda: save_link_wav(state))
    link_dir_btn.pack(side=tk.RIGHT, padx=(5, 0))


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
    subsonic_freq_spinbox = tk.Spinbox(param_frame, textvariable=subsonic_freq, from_=1.0, to=20.0, increment=0.5, width=6, format="%.1f")
    subsonic_freq_spinbox.pack(side=tk.RIGHT, padx=(5, 0))
    subsonic_checkbox = ttk.Checkbutton(param_frame, text="SubSonic", variable=subsonic)
    subsonic_checkbox.pack(side=tk.RIGHT, padx=(10, 0))

    columns = ('Type', 'Frequency', 'Gain', 'Q')
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    tree.pack(fill=tk.BOTH, expand=True)    # --- Словарь состояния для передачи в main.py ---
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
        subsonic_freq=subsonic_freq,
        ir_save_dir=ir_save_dir,
        tree=tree,
        path_label=path_label,
        ir_path_label=ir_path_label,
        override_name=override_name,
        link_ir=link_ir,
        link_wav_dir=link_wav_dir,
        link_path_label=link_path_label,
        # --- добавлено для множественного экспорта ---
        fs_var_list=fs_var_list,
        fs_options=fs_options,
        ir_type_var_list=ir_type_var_list,
        ir_type_options=ir_type_options
    )
    
    # --- Привязки событий ---
    path_label.bind('<Configure>', lambda e: path_label.config(text=truncate_middle(save_dir.get(), path_label.winfo_width() // 7)))
    ir_path_label.bind('<Configure>', lambda e: ir_path_label.config(text=truncate_middle(ir_save_dir.get(), ir_path_label.winfo_width() // 7)))
    link_path_label.bind('<Configure>', lambda e: link_path_label.config(text=truncate_middle(link_wav_dir.get(), link_path_label.winfo_width() // 7)))
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', lambda event: on_drop(event, state))

    def on_close():
        from squig2proq.main import save_window_position, save_config
        save_window_position(root)
        # --- Сохраняем выбранные значения fs и ir_type ---
        fs_selected = [fs for fs, v in zip(fs_options, fs_var_list) if v.get()]
        ir_type_selected = [t for t, v in zip(ir_type_options, ir_type_var_list) if v.get()]
        save_config(
            last_file=current_file.get(),
            last_file_name=file_name.get(),
            export_dir=save_dir.get(),
            ir_export_dir=ir_save_dir.get(),
            adjust_q=adjust_q.get(),
            ir_type=ir_type.get(),
            ir_fs=ir_fs.get(),
            preamp=preamp.get(),
            tilt=tilt.get(),
            subsonic=subsonic.get(),
            subsonic_freq=subsonic_freq.get(),
            override_name=override_name.get(),
            link_wav_dir=link_wav_dir.get(),
            link_ir=link_ir.get(),
            fs_selected=fs_selected,
            ir_type_selected=ir_type_selected
        )
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # --- Запуск ---
    if current_file.get():
        try:
            load_from_path(current_file.get(), False, state)
        except Exception:
            pass
    root.mainloop()

def main():
    launch_gui()

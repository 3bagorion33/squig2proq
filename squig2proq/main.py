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
from squig2proq.ir_utils import fir_from_peq_linear_phase, impulse_response_from_peq, save_ir_to_wav
from squig2proq.ir_utils import fir_from_peq_mixed_phase, apply_tilt_conv_min_phase, apply_tilt_conv_linear_phase, apply_tilt

def load_config():
    # ...existing code from gui.py...
    CONFIG_DIR = Path.home() / ".squig2proq"
    CONFIG_PATH = CONFIG_DIR / "config.json"
    if (CONFIG_PATH.exists()):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Error reading config:", e)
    return {
        "last_file": "",
        "export_dir": "",
        "last_file_name": "",
        "adjust_q": False,
        "ir_type": "Linear Phase",
        "ir_fs": 48000,
        "ir_export_dir": ""
    }

def save_config(**kwargs):
    # ...existing code from gui.py...
    CONFIG_DIR = Path.home() / ".squig2proq"
    CONFIG_PATH = CONFIG_DIR / "config.json"
    config = load_config()
    config.update(kwargs)
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except Exception as e:
        print("Error saving config:", e)

def get_default_install_path():
    # ...existing code from gui.py...
    SHGFP_TYPE_CURRENT = 0
    CSIDL_PERSONAL = 5  # "My Documents"
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    if ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf) != 0:
        raise OSError("Cannot find Documents folder")
    documents = Path(buf.value) / "FabFilter" / "Presets" / "Pro-Q 4"
    documents.mkdir(parents=True, exist_ok=True)
    return str(documents)

# Все функции-обработчики и логика, которые были в launch_gui, теперь здесь:

def load_from_path(path, update_file_name, state):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        filters, preamp_val = parse_filter_text(content)
        state['filters'] = filters
        state['preamp'].set(preamp_val)
        for row in state['tree'].get_children():
            state['tree'].delete(row)
        for f in filters:
            state['tree'].insert('', tk.END, values=(f['Type'], f['Frequency'], f['Gain'], f['Q']))
        state['current_file'].set(path)
        state['status_var'].set(f"Loaded file: {path}")
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get()
        )
    except Exception as e:
        state['status_var'].set(f"Error loading file: {e}")

def load_file(state):
    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")],
        initialdir=Path(state['current_file'].get()).parent if state['current_file'].get() else None
    )
    if file_path:
        load_from_path(file_path, True, state)

def choose_file(state):
    dir_path = filedialog.askdirectory(initialdir=state['save_dir'].get())
    if dir_path:
        state['save_dir'].set(str(dir_path))
        state['path_label'].config(text=truncate_middle(dir_path))
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get()
        )

def export_ffp(state):
    filters = state.get('filters', [])
    if not filters:
        state['status_var'].set("No filters to export.")
        return
    try:
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(base_path, "squig2proq", "data", "Clean.ffp")
        if not os.path.exists(template_path):
            template_path = os.path.join(base_path, "data", "Clean.ffp")
        with open(template_path, 'r', encoding='utf-8') as template_file:
            template_text = template_file.read()
    except Exception as e:
        state['status_var'].set(f"Template read error: {e}")
        return
    result = build_ffp(filters, template_text, adjust_q=state['adjust_q'].get(), preamp=state['preamp'].get())
    filename = state['file_name'].get().strip()
    directory = state['save_dir'].get().strip()
    if not filename:
        state['status_var'].set("Filename is empty")
        return
    if not directory:
        state['status_var'].set("Directory not selected")
        return
    output_path = Path(directory) / f"{filename}.ffp"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(result)
        state['status_var'].set(f"Saved to {output_path}")
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get()
        )
    except Exception as e:
        state['status_var'].set(f"Error saving file: {e}")

def on_drop(event, state):
    try:
        path = state['root'].tk.splitlist(event.data)[0]
        if path.lower().endswith(".txt"):
            load_from_path(path, True, state)
    except Exception as e:
        state['status_var'].set(f"Drag-and-drop error: {e}")

def save_ir_wav(state):
    # Оставляем только выбор директории, убираем всю логику сохранения файла
    dir_path = filedialog.askdirectory(initialdir=state['ir_save_dir'].get())
    if (dir_path):
        state['ir_save_dir'].set(str(dir_path))
        # Обновляем текст ir_path_label после выбора директории
        if 'ir_path_label' in state:
            state['ir_path_label'].config(text=truncate_middle(dir_path))
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get()
        )
        state['status_var'].set(f"Директория для IR выбрана: {dir_path}")

def export_ir(state):
    from squig2proq.ir_utils import fir_from_peq_linear_phase, impulse_response_from_peq, save_ir_to_wav
    from squig2proq.ir_utils import fir_from_peq_mixed_phase
    filters = state.get('filters', [])
    if not filters:
        state['status_var'].set("No filters to export.")
        return
    fs = state['ir_fs'].get()
    length = 65536  # Используем четную длину для обоих типов фазы
    phase = state['ir_type'].get()
    tilt_db = state['tilt'].get()
    preamp_db = state['preamp'].get()
    if state['ir_type'].get() == "Minimum Phase":
        fir = impulse_response_from_peq(filters, fs, length, preamp=preamp_db, tilt=tilt_db)
    elif state['ir_type'].get() == "Linear Phase":
        fir = fir_from_peq_linear_phase(filters, fs, length, preamp=preamp_db, tilt=tilt_db)
    elif state['ir_type'].get() == "Mixed Phase":
        fir = fir_from_peq_mixed_phase(filters, fs, preamp=preamp_db, tilt=tilt_db)
    else:
        state['status_var'].set("Unknown IR type selected.")
        return
    
    filename = state['file_name'].get() or "IR"
    out_path = Path(state['ir_save_dir'].get()) / f"{filename} {phase} {fs}Hz.wav"
    try:
        save_ir_to_wav(fir, str(out_path), fs)
        state['status_var'].set(f"Saved IR to {out_path}")
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get()
        )
    except Exception as e:
        state['status_var'].set(f"Error saving IR: {e}")

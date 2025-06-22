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
import ctypes

def get_window_rect(hwnd):
    # Получить глобальные координаты окна через WinAPI
    rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

def get_virtual_screen_size():
    # Получить размеры виртуального рабочего стола (все мониторы)
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    user32 = ctypes.windll.user32
    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return x, y, w, h

def load_config():
    CONFIG_DIR = Path.home() / ".squig2proq"
    CONFIG_PATH = CONFIG_DIR / "config.json"
    if (CONFIG_PATH.exists()):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # Проверяем координаты окна относительно виртуального экрана
            x, y = config.get('window_x'), config.get('window_y')
            w, h = config.get('window_width'), config.get('window_height')
            if x is not None and y is not None and w is not None and h is not None:
                try:
                    vx, vy, vw, vh = get_virtual_screen_size()
                    # Если окно вне виртуального экрана — сбрасываем
                    if not (vx <= x <= vx+vw-50 and vy <= y <= vy+vh-50):
                        config['window_x'] = None
                        config['window_y'] = None
                        config['window_width'] = None
                        config['window_height'] = None
                except Exception:
                    config['window_x'] = None
                    config['window_y'] = None
                    config['window_width'] = None
                    config['window_height'] = None
            # --- Добавляем значения по умолчанию для множественного выбора ---
            if 'fs_selected' not in config:
                config['fs_selected'] = [config.get('ir_fs', 48000)]
            if 'ir_type_selected' not in config:
                config['ir_type_selected'] = [config.get('ir_type', 'Linear Phase')]
            return config
        except Exception as e:
            print("Error reading config:", e)
    return {
        "last_file": "",
        "export_dir": "",
        "last_file_name": "",
        "adjust_q": False,
        "ir_type": "Linear Phase",
        "ir_fs": 48000,
        "ir_export_dir": "",
        "window_x": None,        
        "window_y": None,
        "window_width": None,
        "window_height": None,
        "link_ir": False,
        "link_wav_dir": "",
        "fs_selected": [48000],
        "ir_type_selected": ["Linear Phase"]
    }

def save_config(**kwargs):
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
        if state['preamp'].get() == 0.0:
            state['preamp'].set(preamp_val)
        for row in state['tree'].get_children():
            state['tree'].delete(row)
        for f in filters:
            state['tree'].insert('', tk.END, values=(f['Type'], f['Frequency'], f['Gain'], f['Q']))
        state['current_file'].set(path)
        state['status_var'].set(f"Loaded file: {path}")

        # Новая логика для override_name
        if state['override_name'].get():
            file_name = Path(path).stem
            state['file_name'].set(file_name)

        save_config(
            last_file=state['current_file'].get(),
            last_file_name=state['file_name'].get(),
            preamp=state['preamp'].get(),
            link_ir=state['link_ir'].get(),
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
            subsonic=state['subsonic'].get(),
            subsonic_freq=state['subsonic_freq'].get(),
            link_ir=state['link_ir'].get(),
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
            subsonic=state['subsonic'].get(),
            subsonic_freq=state['subsonic_freq'].get(),
            link_ir=state['link_ir'].get(),
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
            subsonic=state['subsonic'].get(),
            subsonic_freq=state['subsonic_freq'].get(),
            link_ir=state['link_ir'].get(),
        )
        state['status_var'].set(f"Директория для IR выбрана: {dir_path}")

def export_ir(state):
    from squig2proq.ir_utils import fir_from_peq_linear_phase, fir_from_peq_min_phase, save_ir_to_wav
    from squig2proq.ir_utils import fir_from_peq_mixed_phase
    import os
    filters = state.get('filters', [])
    if not filters:
        state['status_var'].set("No filters to export.")
        return
    # --- Получаем списки выбранных fs и ir_type, если есть ---
    fs_list = None
    ir_type_list = None
    if 'fs_var_list' in state and 'fs_options' in state:
        fs_list = [fs for fs, v in zip(state['fs_options'], state['fs_var_list']) if v.get()]
    if 'ir_type_var_list' in state and 'ir_type_options' in state:
        ir_type_list = [t for t, v in zip(state['ir_type_options'], state['ir_type_var_list']) if v.get()]
    # Если списки не заданы, используем одиночные значения
    if not fs_list:
        fs_list = [state['ir_fs'].get()]
    if not ir_type_list:
        ir_type_list = [state['ir_type'].get()]
    length = 65536  # Используем четную длину для обоих типов фазы
    tilt_db = state['tilt'].get()
    preamp_db = state['preamp'].get()
    subsonic_val = state['subsonic_freq'].get() if state['subsonic'].get() else None
    filename = state['file_name'].get() or "IR"
    ir_save_dir = Path(state['ir_save_dir'].get())
    results = []
    for phase in ir_type_list:
        for fs in fs_list:
            if phase == "Minimum Phase":
                fir = fir_from_peq_min_phase(filters, fs, length, preamp=preamp_db, tilt=tilt_db, fade=True, subsonic=subsonic_val)
            elif phase == "Linear Phase":
                fir = fir_from_peq_linear_phase(filters, fs, length, preamp=preamp_db, tilt=tilt_db, fade=True, subsonic=subsonic_val)
            elif phase == "Mixed Phase":
                fir = fir_from_peq_mixed_phase(filters, fs, length, preamp=preamp_db, tilt=tilt_db, fade=True, subsonic=subsonic_val)
            else:
                state['status_var'].set(f"Unknown IR type selected: {phase}.")
                continue
            out_path = ir_save_dir / f"{filename} T{tilt_db} {phase} {fs}Hz.wav"
            try:
                save_ir_to_wav(fir, str(out_path), fs)
                results.append(f"Saved IR to {out_path}")
                # --- Копирование файла (жёсткая ссылка вместо символической), если включена галочка link_ir ---
                if state.get('link_ir') and state['link_ir'].get():
                    link_dir = state.get('link_wav_dir').get() if state.get('link_wav_dir') else None
                    if link_dir:
                        link_name = f"{phase} {fs}Hz.wav"
                        link_path = Path(link_dir) / link_name
                        try:
                            if link_path.exists() or link_path.is_symlink():
                                link_path.unlink()
                            os.link(str(out_path), str(link_path))
                            results[-1] += f" | Жёсткая ссылка создана: {link_path}"
                        except Exception as e:
                            results[-1] += f" | Ошибка создания жёсткой ссылки: {e}"
            except Exception as e:
                results.append(f"Error saving IR: {e}")
    # Сохраняем конфиг после всех экспортов
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
        subsonic=state['subsonic'].get(),
        subsonic_freq=state['subsonic_freq'].get(),
        link_ir=state['link_ir'].get(),
    )
    state['status_var'].set("; ".join(results))

def save_window_position(root):
    try:
        hwnd = root.winfo_id()
        x, y, w, h = get_window_rect(hwnd)
        save_config(window_x=x, window_y=y, window_width=w, window_height=h)
    except Exception as e:
        print(f"Error saving window position: {e}")

def save_link_wav(state):
    dir_path = filedialog.askdirectory(initialdir=state['link_wav_dir'].get())
    if dir_path:
        state['link_wav_dir'].set(str(dir_path))
        # Обновляем текст link_path_label после выбора директории
        if 'link_path_label' in state:
            state['link_path_label'].config(text=truncate_middle(dir_path))
        save_config(
            last_file=state['current_file'].get(),
            export_dir=state['save_dir'].get(),
            last_file_name=state['file_name'].get(),
            adjust_q=state['adjust_q'].get(),
            ir_type=state['ir_type'].get(),
            ir_fs=state['ir_fs'].get(),
            ir_export_dir=state['ir_save_dir'].get(),
            link_wav_dir=state['link_wav_dir'].get(),
            tilt=state['tilt'].get(),
            preamp=state['preamp'].get(),
            subsonic=state['subsonic'].get(),
            subsonic_freq=state['subsonic_freq'].get(),
            link_ir=state['link_ir'].get(),
        )
        state['status_var'].set(f"Директория для WAV выбрана: {dir_path}")

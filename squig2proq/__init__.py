from pathlib import Path

# --- АВТО-КОМПИЛЯЦИЯ .UI В .PY ---
def compile_ui():
    # Пути к файлам
    base_dir = Path(__file__).parent
    ui_file = base_dir / "gui.ui"  # Имя вашего файла дизайна
    py_file = base_dir / "gui.py"  # Имя генерируемого файла

    # Проверяем, изменился ли UI файл
    if not ui_file.exists():
        print(f"Warning: {ui_file} not found!")
        return

    # Если py файла нет или ui новее чем py -> компилируем
    if not py_file.exists() or ui_file.stat().st_mtime > py_file.stat().st_mtime:
        print("Compiling UI...")
        try:
            # Пытаемся использовать PyQt6
            from PyQt6 import uic
            with open(py_file, "w", encoding="utf-8") as fout:
                uic.compileUi(ui_file, fout)
            print("UI Compiled successfully.")
        except ImportError:
            # Если PyQt6 нет, пробуем через os.system (для PySide6 или PyQt5 tools)
            # Раскомментируйте нужную строку ниже, если верхний способ не сработал:
            
            # os.system(f"pyuic5 {ui_file} -o {py_file}")       # Для PyQt5
            # os.system(f"pyside6-uic {ui_file} -o {py_file}") # Для PySide6
            pass

compile_ui()

# --- ЗАПУСК ПРИЛОЖЕНИЯ ---
from squig2proq.__main__ import main

if __name__ == "__main__":
    main()
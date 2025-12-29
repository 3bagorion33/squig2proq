import sys
import json
import os
from pathlib import Path
import ctypes

# Импорты QT
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox, 
                             QTreeWidgetItem, QMenu, QWidgetAction, QCheckBox, 
                             QLabel, QSizePolicy, QHBoxLayout, QSpinBox, QWidget)
from PyQt6.QtGui import QPainter, QGuiApplication
from PyQt6.QtCore import Qt, QSize

# Импорт вашего скомпилированного дизайна
# Убедитесь, что __init__.py отработал и gui.py существует
from squig2proq.gui import Ui_MainWindow 

# Ваши библиотеки логики (оставляем как есть)
from squig2proq.parser import parse_filter_text, build_ffp
from squig2proq.ir_utils import IRCreator, save_ir_to_wav

class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        # Говорим лейауту: "Мне все равно, какой длины текст, дай мне сколько не жалко места"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def setText(self, text):
        # Сохраняем полный текст
        self._full_text = text
        # Вызываем базовый метод (для корректной работы accessibility и прочего)
        super().setText(text)
        # Ставим тултип
        self.setToolTip(text)
        # Принудительно перерисовываем
        self.update()

    def paintEvent(self, event):
        # В момент отрисовки берем полный текст и обрезаем под текущую ширину
        painter = QPainter(self)
        metrics = self.fontMetrics()
        elided = metrics.elidedText(self.text(), Qt.TextElideMode.ElideMiddle, self.width())
        
        # Рисуем текст (с выравниванием, как было настроено в дизайнере)
        # rect() - это вся доступная область лейбла
        painter.drawText(self.rect(), self.alignment(), elided)

    def minimumSizeHint(self):
        # Разрешаем лейблу сжиматься до 0 по ширине
        return QSize(0, super().minimumSizeHint().height())

class SquigApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Загрузка дизайна
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --- ЗАМЕНА ЛЕЙБЛОВ НА УМНЫЕ ---
        # Перезаписываем self.ui.path_label ссылкой на новый виджет
        self.ui.path_label = self.promote_label(self.ui.path_label)
        self.ui.ir_path_label = self.promote_label(self.ui.ir_path_label)
        self.ui.link_path_label = self.promote_label(self.ui.link_path_label)

        # 2. Настройка Drag & Drop (Включено в свойствах дизайнера, но логику пишем тут)
        self.setAcceptDrops(True)

        # 3. Внутренние переменные (вместо tk.StringVar)
        self.config_dir = Path.home() / ".squig2proq"
        self.config_path = self.config_dir / "config.json"
        self.current_file_path = ""
        self.parsed_filters = []
        
        # Списки опций для меню
        self.fs_options = [44100, 48000, 88200, 96000, 176400, 192000]
        self.ir_type_options = ["Linear Phase", "Minimum Phase", "Mixed Phase"]
        self.mixed_phase_ratio = 16
        
        # Переменные для хранения выбора в меню (sets)
        self.selected_fs = {48000}
        self.selected_ir_types = {"Linear Phase"}

        # 4. Инициализация UI (меню, привязки кнопок)
        self.setup_connections()
        self.setup_custom_menus()
        
        # 5. Загрузка конфига и восстановление состояния
        self.load_config()

        # Показываем статус
        self.ui.statusbar.showMessage("Ready")

    def setup_connections(self):
        """Привязываем кнопки к функциям"""
        self.ui.load_btn.clicked.connect(self.browse_file)
        self.ui.ffp_dir_btn.clicked.connect(lambda: self.choose_dir(self.ui.path_label, "export_dir"))
        self.ui.ffp_export_btn.clicked.connect(self.export_ffp)
        
        self.ui.ir_dir_btn.clicked.connect(lambda: self.choose_dir(self.ui.ir_path_label, "ir_export_dir"))
        self.ui.ir_export_btn.clicked.connect(self.export_ir)
        
        self.ui.link_dir_btn.clicked.connect(lambda: self.choose_dir(self.ui.link_path_label, "link_wav_dir"))
        
        # Пример авто-обрезки пути при ресайзе (Qt делает это сам через elide, 
        # но для Label нужно обновлять текст, пока оставим просто полный текст)

    def setup_custom_menus(self):
        """Создаем выпадающие меню с чекбоксами для кнопок"""
        # Меню для Sample Rate
        self.fs_menu = QMenu(self)
        self.ui.fs_btn.setMenu(self.fs_menu)
        
        # Меню для IR Type
        self.ir_type_menu = QMenu(self)
        self.ui.ir_type_btn.setMenu(self.ir_type_menu)

    def populate_menu(self, menu, options, selected_set, btn_widget, label_prefix):
        """
        Заполняет меню. Для 'Mixed Phase' добавляет SpinBox.
        """
        menu.clear()
        
        for opt in options:
            action = QWidgetAction(menu)
            
            # 1. Создаем контейнер и лейаут
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(5, 2, 5, 2) # Немного отступов, чтобы было красиво
            
            # 2. Создаем чекбокс
            cb = QCheckBox(str(opt))
            cb.setChecked(opt in selected_set)
            
            # Логика чекбокса (как и была)
            def on_check(state, val=opt, checkbox=cb):
                if state:
                    selected_set.add(val)
                else:
                    selected_set.discard(val)
                self.update_btn_text(btn_widget, label_prefix, selected_set)

            cb.stateChanged.connect(on_check)
            
            # Добавляем чекбокс в лейаут
            layout.addWidget(cb)
            
            # 3. СПЕЦИАЛЬНАЯ ЛОГИКА ДЛЯ Mixed Phase
            if opt == "Mixed Phase":
                lbl = QLabel("HF Ratio:")
                # Создаем спинбокс
                spin = QSpinBox()
                spin.setRange(8, 256)
                spin.setSingleStep(8)
                spin.setPrefix("1/")
                spin.setValue(int(self.mixed_phase_ratio)) # Текущее значение
                spin.setFixedWidth(52) # Фиксированная ширина, чтобы не прыгало
                
                # Блокируем спинбокс, если чекбокс выключен (опционально)
                is_checked = cb.isChecked()
                spin.setEnabled(is_checked)
                lbl.setEnabled(is_checked)
                cb.stateChanged.connect(lambda state: spin.setEnabled(state))
                cb.stateChanged.connect(lambda state: lbl.setEnabled(state))

                # Сохраняем значение при изменении
                def on_spin_change(val):
                    self.mixed_phase_ratio = val
                spin.valueChanged.connect(on_spin_change)

                # ХАК: Если юзер ввел "7" руками и нажал Enter -> округляем до 8
                def snap_to_8():
                    val = spin.value()
                    remainder = val % 8
                    if remainder != 0:
                        # Округляем до ближайшего кратного 8
                        if remainder >= 4:
                            new_val = val + (8 - remainder)
                        else:
                            new_val = val - remainder
                        spin.setValue(new_val)
                
                # editingFinished срабатывает при нажатии Enter или потере фокуса
                spin.editingFinished.connect(snap_to_8)
                
                # Добавляем спинбокс в лейаут справа
                layout.addWidget(lbl)
                layout.addWidget(spin)
            
            # Растягивающий элемент, если нужно прижать спинбокс вправо (опционально)
            # layout.addStretch() 

            # 4. Добавляем готовый контейнер в меню
            action.setDefaultWidget(container)
            menu.addAction(action)
        
        self.update_btn_text(btn_widget, label_prefix, selected_set)

    def update_btn_text(self, btn, prefix, selected_set):
        if not selected_set:
            btn.setText(prefix)
        else:
            # Сортируем для красоты
            sorted_vals = sorted(list(selected_set), key=lambda x: str(x))
            btn.setText(", ".join(map(str, sorted_vals)))

    def ensure_window_visible(self):
        """
        Проверяет, видно ли окно на экранах.
        Если заголовок окна улетел в космос — возвращает его на землю.
        """
        # Получаем геометрию рамки окна (включая заголовок)
        frame_geo = self.frameGeometry()
        
        # Проверяем, находится ли левый верхний угол окна на каком-либо экране
        screen = QGuiApplication.screenAt(frame_geo.topLeft())
        
        # Если screen is None, значит точка находится в пустоте (за пределами экранов)
        if screen is None:
            print("Окно обнаружено за пределами экрана! Возвращаем на базу.")
            # Берем основной монитор
            primary_screen = QGuiApplication.primaryScreen()
            available_geo = primary_screen.availableGeometry()
            
            # Вычисляем центр
            center_point = available_geo.center()
            frame_geo.moveCenter(center_point)
            
            # Применяем новые координаты (move перемещает верхний левый угол)
            self.move(frame_geo.topLeft())

    # --- ЛОГИКА КОНФИГА ---
    def load_config(self):
        defaults = {
            "window_width": 800, "window_height": 700,
            "fs_base": 48000, "preamp": 0.0, "tilt": 0.0,
            "subsonic": True, "subsonic_freq": 10.0,
            "adjust_q": False, "override_name": False, "link_ir": False
        }
        
        config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

        # Восстановление геометрии окна
        if all(k in config for k in ("window_x", "window_y", "window_width", "window_height")):
            try:
                # 1. Сначала задаем размер внутренней части (width/height из конфига — это клиенская часть)
                self.resize(int(config["window_width"]), int(config["window_height"]))
                # 2. Потом двигаем само окно по координатам (x/y из конфига — это позиция рамки)
                self.move(int(config["window_x"]), int(config["window_y"]))
            except: pass
        
        # Заполнение полей UI
        self.ui.fs_base_entry.setText(str(config.get("fs_base", defaults["fs_base"])))
        self.ui.preamp_spinbox.setValue(float(config.get("preamp", defaults["preamp"])))
        self.ui.tilt_spinbox.setValue(float(config.get("tilt", defaults["tilt"])))
        self.ui.subsonic_checkbox.setChecked(config.get("subsonic", defaults["subsonic"]))
        self.ui.subsonic_freq_spinbox.setValue(float(config.get("subsonic_freq", defaults["subsonic_freq"])))
        self.ui.adjust_q_checkbox.setChecked(config.get("adjust_q", defaults["adjust_q"]))
        self.ui.override_checkbox.setChecked(config.get("override_name", defaults["override_name"]))
        self.ui.link_ir_checkbox.setChecked(config.get("link_ir", defaults["link_ir"]))

        self.mixed_phase_ratio = int(round(config.get("mixed_phase_ratio", 16) / 8) * 8)
        
        # Пути
        def_path = self.get_default_install_path()
        self.export_dir = config.get("export_dir") or def_path
        self.ir_export_dir = config.get("ir_export_dir") or self.export_dir
        self.link_wav_dir = config.get("link_wav_dir") or self.export_dir
        self.current_file_path = config.get("last_file", "")
        self.ui.name_entry.setText(config.get("last_file_name", "Parsed_Filter"))

        # Обновляем лейблы
        self.ui.path_label.setText(self.export_dir)
        self.ui.ir_path_label.setText(self.ir_export_dir)
        self.ui.link_path_label.setText(self.link_wav_dir)

        # Восстанавливаем выбор в меню
        if "fs_selected" in config:
            self.selected_fs = set(config["fs_selected"])
        if "ir_type_selected" in config:
            self.selected_ir_types = set(config["ir_type_selected"])

        # Перестраиваем меню с учетом загруженных данных
        self.populate_menu(self.fs_menu, self.fs_options, self.selected_fs, self.ui.fs_btn, "Sample Rate")
        self.populate_menu(self.ir_type_menu, self.ir_type_options, self.selected_ir_types, self.ui.ir_type_btn, "IR Type")

        # Если был последний файл, пробуем загрузить
        if self.current_file_path and os.path.exists(self.current_file_path):
            self.load_from_path(self.current_file_path, update_name=False)

        self.ensure_window_visible()

    def save_config(self):
        config = {
            "window_x": self.x(),
            "window_y": self.y(),
            "window_width": self.width(),
            "window_height": self.height(),
            
            "last_file": self.current_file_path,
            "last_file_name": self.ui.name_entry.text(),
            
            "export_dir": self.export_dir,
            "ir_export_dir": self.ir_export_dir,
            "link_wav_dir": self.link_wav_dir,
            
            "fs_base": int(self.ui.fs_base_entry.text() or 48000),
            "preamp": self.ui.preamp_spinbox.value(),
            "tilt": self.ui.tilt_spinbox.value(),
            
            "subsonic": self.ui.subsonic_checkbox.isChecked(),
            "subsonic_freq": self.ui.subsonic_freq_spinbox.value(),
            "adjust_q": self.ui.adjust_q_checkbox.isChecked(),
            "override_name": self.ui.override_checkbox.isChecked(),
            "link_ir": self.ui.link_ir_checkbox.isChecked(),
            
            "fs_selected": list(self.selected_fs),
            "ir_type_selected": list(self.selected_ir_types),
            "mixed_phase_ratio": self.mixed_phase_ratio
        }
        
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    # --- ФУНКЦИОНАЛ ---
    def get_default_install_path(self):
        # Оставил вашу логику поиска папки
        try:
            SHGFP_TYPE_CURRENT = 0
            CSIDL_PERSONAL = 5 
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
            documents = Path(buf.value) / "FabFilter" / "Presets" / "Pro-Q 4"
            documents.mkdir(parents=True, exist_ok=True)
            return str(documents)
        except:
            return str(Path.home())

    def choose_dir(self, label_widget, attr_name):
        initial = getattr(self, attr_name, "")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", initial)
        if dir_path:
            setattr(self, attr_name, dir_path)
            label_widget.setText(dir_path)
            self.save_config()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load AutoEQ File", 
            os.path.dirname(self.current_file_path) if self.current_file_path else "",
            "Text files (*.txt)"
        )
        if file_path:
            self.load_from_path(file_path, update_name=True)

    def load_from_path(self, path, update_name=True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.parsed_filters, preamp_val, fs_base = parse_filter_text(content)
            
            # Обновляем UI
            self.ui.fs_base_entry.setText(str(fs_base if fs_base else 48000))
            
            # Preamp: если стоит 0, берем из файла
            if self.ui.preamp_spinbox.value() == 0.0:
                self.ui.preamp_spinbox.setValue(float(preamp_val))
            
            # Заполняем таблицу
            self.ui.tree.clear()
            for f in self.parsed_filters:
                item = QTreeWidgetItem(self.ui.tree)
                item.setText(0, str(f['Type']))
                item.setText(1, str(f['Frequency']))
                item.setText(2, str(f['Gain']))
                item.setText(3, str(f['Q']))
            
            self.current_file_path = path
            self.ui.statusbar.showMessage(f"Loaded: {Path(path).name}")

            if self.ui.override_checkbox.isChecked() or update_name:
                self.ui.name_entry.setText(Path(path).stem)
                
            self.save_config()

        except Exception as e:
            self.ui.statusbar.showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    # --- DRAG AND DROP ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(".txt"):
                self.load_from_path(f, update_name=True)
                break # Загружаем только первый файл

    # --- ЭКСПОРТ ---
    def export_ffp(self):
        if not self.parsed_filters:
            self.ui.statusbar.showMessage("No filters loaded.")
            return

        # Логика поиска шаблона
        try:
            # Определяем базовый каталог
            if getattr(sys, "frozen", False):
                # В Nuitka --onefile или standalone временный каталог
                base_path = Path(sys.executable).resolve().parent
            else:
                base_path = Path(__file__).resolve().parent

            # Составляем возможные пути к файлу
            possible_paths = [
                base_path / "squig2proq" / "data" / "Clean.ffp",
                base_path / "data" / "Clean.ffp",
                base_path / "Clean.ffp"
            ]

            # Находим существующий
            template_path = next((p for p in possible_paths if p.exists()), None)
            if template_path is None:
                raise FileNotFoundError("Clean.ffp not found in expected locations")

            # Читаем файл
            with open(template_path, "r", encoding="utf-8") as tf:
                template_text = tf.read()

        except Exception as e:
            self.ui.statusbar.showMessage(f"Template error: {e}")
            return

        # Сборка данных
        result = build_ffp(
            self.parsed_filters, 
            template_text, 
            adjust_q=self.ui.adjust_q_checkbox.isChecked(), 
            preamp=self.ui.preamp_spinbox.value()
        )
        
        fname = self.ui.name_entry.text().strip() or "Filter"
        out_path = Path(self.export_dir) / f"{fname}.ffp"
        
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(result)
            self.ui.statusbar.showMessage(f"Saved: {out_path.name}")
        except Exception as e:
            self.ui.statusbar.showMessage(f"Error saving: {e}")

    def export_ir(self):
        if not self.parsed_filters:
            self.ui.statusbar.showMessage("No filters loaded.")
            return

        length = 65536
        tilt_db = self.ui.tilt_spinbox.value()
        preamp_db = self.ui.preamp_spinbox.value()
        fs_base_val = int(self.ui.fs_base_entry.text() or 48000)
        subsonic_val = self.ui.subsonic_freq_spinbox.value() if self.ui.subsonic_checkbox.isChecked() else None
        
        filename = self.ui.name_entry.text() or "IR"
        ir_dir = Path(self.ir_export_dir)
        
        ircreator = IRCreator(self.parsed_filters, preamp=preamp_db, fs_base=fs_base_val, subsonic=subsonic_val, tilt=tilt_db)
        
        results_msg = []
        
        for phase in self.selected_ir_types:
            for fs in self.selected_fs:
                try:
                    if phase == "Minimum Phase":
                        fir = ircreator.min_phase(fs, length, fade=True)
                    elif phase == "Linear Phase":
                        fir = ircreator.lin_phase(fs, length, fade=True)
                    elif phase == "Mixed Phase":
                        fir = ircreator.mix_phase(fs, length, fade=True, mix_ratio=self.mixed_phase_ratio)
                    else: continue
                    
                    name_suffix = f"{filename} T{tilt_db} {phase} {fs}Hz.wav"
                    out_path = ir_dir / name_suffix
                    
                    save_ir_to_wav(fir, str(out_path), fs)
                    results_msg.append(f"Ok: {fs}Hz")

                    # Hard Link
                    if self.ui.link_ir_checkbox.isChecked() and self.link_wav_dir:
                        link_name = f"{phase} {fs}Hz.wav"
                        link_path = Path(self.link_wav_dir) / link_name
                        if link_path.exists() or link_path.is_symlink():
                            try: link_path.unlink() 
                            except: pass
                        try:
                            os.link(str(out_path), str(link_path))
                        except Exception as e:
                            print(f"Link error: {e}")

                except Exception as e:
                    print(f"IR Error {phase} {fs}: {e}")
        
        self.ui.statusbar.showMessage(f"Exported {len(results_msg)} IRs.")
        self.save_config()

    def closeEvent(self, event):
        """Событие закрытия окна"""
        self.save_config()
        event.accept()

    def promote_label(self, ui_label):
        """
        Заменяет обычный QLabel из UI на наш ElidedLabel,
        сохраняя все настройки (шрифт, позицию, лейаут).
        """
        parent = ui_label.parentWidget()
        layout = parent.layout()
        
        # Создаем наш умный лейбл с текстом старого
        new_label = ElidedLabel(ui_label.text(), parent)
        
        # Копируем свойства
        new_label.setFont(ui_label.font())
        new_label.setAlignment(ui_label.alignment())
        new_label.setStyleSheet(ui_label.styleSheet())
        # new_label.setGeometry(ui_label.geometry()) # Не обязательно, лейаут сам решит
        
        # Заменяем в лейауте старый на новый
        layout.replaceWidget(ui_label, new_label)
        
        # Удаляем старый
        ui_label.deleteLater()
        
        return new_label

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SquigApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
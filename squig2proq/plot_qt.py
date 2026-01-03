import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from dataclasses import dataclass
from typing import List, Callable, Optional

# --- Вспомогательные классы (как были в старой версии) ---
@dataclass
class MouseEventData:
    """Структура данных для события движения мыши"""
    x: float
    y: float
    x_pixel: float
    y_pixel: float
    button_mask: int
    # Количество шагов колеса (положительное/отрицательное) если доступно
    step: Optional[float] = None

class Event:
    """Класс для реализации событий (C#-style)"""
    def __init__(self):
        self._handlers: List[Callable] = []
    
    def __iadd__(self, handler: Callable):
        self._handlers.append(handler)
        return self
    
    def __isub__(self, handler: Callable):
        if handler in self._handlers:
            self._handlers.remove(handler)
        return self
    
    def __call__(self, *args, **kwargs):
        for handler in self._handlers:
            handler(*args, **kwargs)

class PlotWidget(QWidget):
    """
    Виджет графика PyQt6 + Matplotlib.
    Наследуется от QWidget, чтобы его можно было легко вставить в Layout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. Настройка Layout для самого виджета
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) # Убираем отступы
        
        # 2. Инициализация переменных состояния
        self.mouse_pressed = False
        self.background = None
        self._current_button_mask = 0
        
        # События (для подписки извне)
        self.mouse_move = Event()
        self.mouse_press = Event()
        self.mouse_release = Event()
        self.mouse_wheel = Event()
        
        # 3. Создаем фигуру Matplotlib
        self.fig, self.ax = self._create_plot()
        
        # 4. Создаем Canvas (холст), связывающий Matplotlib и Qt
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        
        # Добавляем Canvas внутрь нашего QWidget
        self.layout.addWidget(self.canvas)
        
        # 5. Начальная настройка осей
        self.ax.set_ylim(-12, 12)
        self.ax.margins(x=0, y=0)
        
        # Вертикальная линия (курсор)
        self.vline = self.ax.axvline(x=1000, color='green', linestyle='-', alpha=0.5)
        
        # 6. Подключаем события Matplotlib
        self._setup_event_handlers()

    def _create_plot(self):
        """Создает фигуру (настройки внешнего вида графика)"""
        fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        # Немного места снизу для подписей
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.05) 
        
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        
        ax.set_xscale('log')
        ax.set_xlim(20, 20000)
        
        # Ось Y
        ax.set_yticks([-10, 0, 10])
        ax.set_yticklabels(['-10', '0', '+10'], color='gray')
        ax.set_yticks([-15, -5, 5, 15], minor=True)
        
        # Ось X
        ax.set_xticks([100, 1000, 10000])
        ax.set_xticklabels(['100', '1k', '10k'])
        
        # Убираем рамку
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Стиль подписей
        ax.tick_params(axis='x', which='major', labelsize=8, pad=5, colors='gray')
        
        # Сетка
        ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5, axis='both')
        
        # Минорная сетка по X
        minor_ticks_x = []
        for decade in [10, 100, 1000, 10000]:
            for i in range(1, 10, 1):
                freq = decade * i
                if 20 <= freq <= 20000:
                    minor_ticks_x.append(freq)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.grid(True, which='minor', color='gray', linestyle=':', alpha=0.3, axis='both')
        
        return fig, ax

    def _setup_event_handlers(self):
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    # --- Обработчики событий (почти копия из plot.py) ---
    def _on_mouse_press(self, event):
        self._update_button_mask(event.button, True)
        if event.inaxes:
            data = self._make_event_data(event)
            self.mouse_press(data)
        
        if event.button == 3: # Правая кнопка
            self.mouse_pressed = True
            self.vline.set_visible(False)
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.vline.set_visible(True)
            self._on_mouse_move(event)

    def _on_mouse_release(self, event):
        self._update_button_mask(event.button, False)
        if event.inaxes:
            data = self._make_event_data(event)
            self.mouse_release(data)
        if event.button == 3:
            self.mouse_pressed = False

    def _on_mouse_move(self, event):
        if event.inaxes:
            data = self._make_event_data(event)
            self.mouse_move(data)
        
        if self.mouse_pressed and event.inaxes:
            x = event.xdata
            if x is not None:
                if self.background is not None:
                    self.canvas.restore_region(self.background)
                self.vline.set_xdata([x, x])
                self.ax.draw_artist(self.vline)
                self.canvas.blit(self.ax.bbox)

    def _make_event_data(self, event):
        step = getattr(event, 'step', None)
        return MouseEventData(
            x=event.xdata, y=event.ydata,
            x_pixel=event.x, y_pixel=event.y,
            button_mask=self._current_button_mask,
            step=step
        )

    def _on_scroll(self, event):
        """Обработчик вращения колесика мыши над графиком"""
        if event.inaxes:
            data = self._make_event_data(event)
            # Для совместимости: если step не задан, попробуем вывести направление из event.button
            if data.step is None:
                btn = getattr(event, 'button', None)
                if btn == 'up':
                    data.step = 1
                elif btn == 'down':
                    data.step = -1
            self.mouse_wheel(data)

    def _update_button_mask(self, button, pressed):
        if button is None: return
        button_bit = 1 << (button - 1)
        if pressed:
            self._current_button_mask |= button_bit
        else:
            self._current_button_mask &= ~button_bit

    # --- Публичные методы ---
    def plot_frequency_response(self, frequencies, magnitudes):
        # Удаляем старые линии (кроме вертикальной и сетки)
        lines_to_remove = [line for line in self.ax.lines if line != self.vline]
        for line in lines_to_remove:
            line.remove()
            
        self.ax.plot(frequencies, magnitudes, color='blue', linewidth=2, label='EQ Response')
        self.canvas.draw()

    def clear_plot(self):
        lines_to_remove = [line for line in self.ax.lines if line != self.vline]
        for line in lines_to_remove:
            line.remove()
        self.canvas.draw()
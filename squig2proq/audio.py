import numpy as np
import threading
import time
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Библиотека sounddevice не установлена.")

class AudioManager:
    """
    Класс для управления аудио.
    НЕ зависит от GUI фреймворка (Qt/Tkinter).
    """
    
    def __init__(self):
        self.devices = []
        self.current_device_id = None
        self.is_playing = False
        self.stream = None
        self.stop_event = threading.Event()
        
        self.current_frequency = 1000.0
        self.target_frequency = 1000.0
        self.frequency_smoothing = 0.85
        
        # Application-level volume (0.0 .. 1.0) and lock for thread safety
        self.volume = 0.2
        self._volume_lock = threading.Lock()
        
        self.mouse_move_subscribed = False
        self.mouse_events_subscribed = False
        self.is_right_mouse_tone_active = False
        self._active_plot_widget = None  # Ссылка на текущий график

        if not SOUNDDEVICE_AVAILABLE:
            return
            
        try:
            sd.check_input_settings()
            sd.check_output_settings()
        except Exception as e:
            print(f"Предупреждение sounddevice: {e}")

    def get_audio_output_devices(self):
        """Возвращает список (id, name)"""
        if not SOUNDDEVICE_AVAILABLE:
            return []
        
        try:
            devices = [(None, "Default Device")]
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_output_channels'] > 0:
                    hostapi = sd.query_hostapis(device['hostapi'])['name']
                    name = f"{device['name']} ({hostapi})"
                    devices.append((i, name))
            self.devices = devices
            return devices
        except Exception as e:
            print(f"Ошибка получения устройств: {e}")
            return [(None, "Default Device")]

    def generate_tone_callback(self, sample_rate, volume):
        """Callback для генерации звука (SoundDevice)"""
        phase_accumulator = 0.0
        
        def callback(outdata, frames, time, status):
            nonlocal phase_accumulator
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            
            # Сглаживание частоты
            freq_diff = abs(self.target_frequency - self.current_frequency)
            if freq_diff > 50.0:
                self.current_frequency = (self.current_frequency * self.frequency_smoothing + 
                                        self.target_frequency * (1 - self.frequency_smoothing))
            else:
                self.current_frequency = self.target_frequency
            
            # Read current volume atomically
            try:
                self._volume_lock
            except AttributeError:
                # In case volume/lock aren't present (very old instance), fall back to parameter
                vol = volume
            else:
                with self._volume_lock:
                    vol = self.volume

            dt = 1.0 / sample_rate
            omega = 2 * np.pi * self.current_frequency
            t_offsets = np.arange(frames) * dt
            phases = phase_accumulator + omega * t_offsets
            tone = vol * np.sin(phases)
            
            phase_accumulator += omega * frames * dt
            while phase_accumulator >= 2 * np.pi:
                phase_accumulator -= 2 * np.pi
            
            outdata[:, 0] = tone
            if outdata.shape[1] > 1:
                outdata[:, 1] = tone
        
        return callback

    def start_tone(self, device_id, frequency=1000.0, volume=0.2):
        """Запуск воспроизведения"""
        if not SOUNDDEVICE_AVAILABLE: return False
        
        self.stop_tone()
        self.current_device_id = device_id
        self.current_frequency = frequency
        self.target_frequency = frequency
        self.stop_event.clear()
        # Apply requested initial volume to application-level volume
        try:
            with self._volume_lock:
                self.volume = max(0.0, min(1.0, float(volume)))
        except Exception:
            # If lock/volume missing for any reason, set attribute directly
            self.volume = max(0.0, min(1.0, float(volume)))
        
        try:
            sr = 48000
            callback = self.generate_tone_callback(sr, volume)
            self.stream = sd.OutputStream(
                device=device_id, samplerate=sr, channels=1,
                callback=callback, blocksize=int(sr * 0.001), latency='low'
            )
            self.stream.start()
            self.is_playing = True
            return True
        except Exception as e:
            print(f"Ошибка старта аудио: {e}")
            self.is_playing = False
            return False

    def stop_tone(self):
        """Остановка воспроизведения"""
        if self.is_playing:
            self.stop_event.set()
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except: pass
                self.stream = None
            self.is_playing = False

    def is_tone_playing(self):
        return self.is_playing and self.stream is not None

    # --- Volume control (application-level) ---
    def set_volume(self, value):
        """Set application volume (0.0..1.0). Thread-safe."""
        try:
            v = float(value)
        except Exception:
            return False
        v = max(0.0, min(1.0, v))
        with self._volume_lock:
            self.volume = v
        return True

    def get_volume(self):
        """Get current application volume (0.0..1.0)."""
        with self._volume_lock:
            return float(self.volume)

    def change_volume(self, delta):
        """Change volume by delta (positive or negative). Returns new volume."""
        try:
            d = float(delta)
        except Exception:
            return None
        with self._volume_lock:
            self.volume = max(0.0, min(1.0, self.volume + d))
            return float(self.volume)

    # --- Slider <-> amplitude mapping (log scale) ---
    def slider_to_amp(self, slider_val: int, slider_min: int = 0, slider_max: int = 100, db_min: float = -60.0) -> float:
        """Map slider position (slider_min..slider_max) to amplitude (0..1) logarithmically.

        Defaults map slider_min -> db_min (e.g. -60 dB), slider_max -> 0 dB.
        """
        try:
            s = int(slider_val)
        except Exception:
            s = slider_min
        s = max(slider_min, min(slider_max, s))
        # linear interpolation in dB domain
        db = db_min + (s - slider_min) / float(slider_max - slider_min) * (0.0 - db_min)
        amp = 10 ** (db / 20.0)
        return float(amp)

    def amp_to_slider(self, amp: float, slider_min: int = 0, slider_max: int = 100, db_min: float = -60.0) -> int:
        """Inverse mapping: amplitude (0..1) to slider position (slider_min..slider_max)."""
        try:
            a = float(amp)
        except Exception:
            return slider_min
        if a <= 0:
            return slider_min
        db = 20.0 * np.log10(a)
        pos = slider_min + (db - db_min) / (0.0 - db_min) * (slider_max - slider_min)
        return max(slider_min, min(slider_max, int(round(pos))))

    def set_volume_from_slider(self, slider_val: int, slider_min: int = 0, slider_max: int = 100, db_min: float = -60.0) -> float:
        """Set application volume from slider position. Returns amplitude."""
        amp = self.slider_to_amp(slider_val, slider_min, slider_max, db_min)
        self.set_volume(amp)
        return amp

    def get_slider_value(self, slider_min: int = 0, slider_max: int = 100, db_min: float = -60.0) -> int:
        """Get slider position corresponding to current application volume."""
        return self.amp_to_slider(self.get_volume(), slider_min, slider_max, db_min)

    # --- Подписки на события мыши ---

    def subscribe_to_mouse_move(self, plot_widget):
        """Подписка только на движение (для кнопки Test Audio)"""
        if self.mouse_move_subscribed: return
        try:
            plot_widget.mouse_move += self._on_mouse_move_event
            self._active_plot_widget = plot_widget
            self.mouse_move_subscribed = True
        except Exception as e:
            print(f"Ошибка подписки: {e}")

    def unsubscribe_mouse_move(self):
        """Отписка от движения"""
        if not self.mouse_move_subscribed or not self._active_plot_widget: return
        try:
            self._active_plot_widget.mouse_move -= self._on_mouse_move_event
            self.mouse_move_subscribed = False
        except: pass

    def subscribe_right_click_tone(self, plot_widget):
        """Подписка на правую кнопку мыши (для режима дизайнера)"""
        if self.mouse_events_subscribed: return
        try:
            plot_widget.mouse_press += self._on_mouse_press_event
            plot_widget.mouse_release += self._on_mouse_release_event
            plot_widget.mouse_move += self._on_mouse_move_for_right_button
            self._active_plot_widget = plot_widget
            self.mouse_events_subscribed = True
        except Exception as e:
            print(f"Ошибка подписки ПКМ: {e}")

    def unsubscribe_right_click_tone(self):
        if not self.mouse_events_subscribed or not self._active_plot_widget: return
        try:
            self._active_plot_widget.mouse_press -= self._on_mouse_press_event
            self._active_plot_widget.mouse_release -= self._on_mouse_release_event
            self._active_plot_widget.mouse_move -= self._on_mouse_move_for_right_button
            self.mouse_events_subscribed = False
        except: pass

    # --- Обработчики событий ---

    def _update_freq_from_x(self, x):
        if x > 0:
            freq = max(20.0, min(20000.0, x))
            self.target_frequency = freq
            if self.is_tone_playing():
                # Если звук играет, обновляем цель, сглаживание сделает callback
                pass

    def _on_mouse_move_event(self, mouse_data):
        self._update_freq_from_x(mouse_data.x)

    def _on_mouse_press_event(self, mouse_data):
        # Если нажата Правая кнопка (mask 4)
        if mouse_data.button_mask & 4:
            if not self.is_right_mouse_tone_active:
                if mouse_data.x > 0:
                    freq = max(20.0, min(20000.0, mouse_data.x))
                    # Use current application volume (may be set from UI slider)
                    current_vol = self.get_volume()
                    if self.start_tone(self.current_device_id, freq, current_vol):
                        self.is_right_mouse_tone_active = True

    def _on_mouse_release_event(self, mouse_data):
        # Если правая кнопка отпущена (нет бита 4) и звук был активен
        if self.is_right_mouse_tone_active and not (mouse_data.button_mask & 4):
            self.stop_tone()
            self.is_right_mouse_tone_active = False

    def _on_mouse_move_for_right_button(self, mouse_data):
        # Обновляем, если правая кнопка зажата и звук активен
        if self.is_right_mouse_tone_active and (mouse_data.button_mask & 4):
            self._update_freq_from_x(mouse_data.x)
        # Если вдруг кнопка отпущена, но событие release пропустили (бывает при выходе за окно)
        elif self.is_right_mouse_tone_active and not (mouse_data.button_mask & 4):
            self.stop_tone()
            self.is_right_mouse_tone_active = False

    def cleanup(self):
        self.stop_tone()
        self.unsubscribe_mouse_move()
        self.unsubscribe_right_click_tone()
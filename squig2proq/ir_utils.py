import numpy as np
from scipy.signal import lfilter, freqz, bessel, oaconvolve
from scipy.io import wavfile
from scipy.fft import dct, idct, fft, ifft

class IRCreator:
    """
    Класс для создания импульсных характеристик (ИХ) на основе параметрических фильтров.
    Поддерживает минимальную фазу, линейную фазу и смешанную фазу.
    """

    def __init__(self, filters, preamp: float=0.0, fs_base: int=48000, subsonic: float=None, tilt: float=0.0):
        """
        Инициализация процессора PEQ.
        
        Parameters
        ----------
        filters : list
            Список параметрических фильтров
        preamp : float, optional
            Предусиление в дБ. По умолчанию 0.0
        fs_base : int, optional
            Базовая частота дискретизации. По умолчанию 48000
        """
        self.filters = filters
        self.fs_base = fs_base
        self.preamp = db_to_amp(preamp)
        self.subsonic = subsonic
        self.tilt = tilt
        self.pivot_hz = 1000.0
        self._coeffs_cache, self._lp_coeffs_cache, self._hp_coeffs_cache = self._calculate_filter_coeffs()

    def _calculate_filter_coeffs(self):
        """
        Вычисляет и кэширует коэффициенты всех фильтров для заданной частоты дискретизации.
        
        Parameters
        ----------
        fs : int
            Частота дискретизации
            
        Returns
        -------
        list
            Список кортежей (b, a) коэффициентов для каждого фильтра
        """
        coeffs = []
        lp_coeffs = []
        hp_coeffs = []
        crossover   = 3000 # частота сшивки в Гц
        xfade_width = 0.2  # ширина кроссовера в октавах
        xfade_low = crossover * 2**(-xfade_width/2)
        xfade_high = crossover * 2**(xfade_width/2)
    
        for f in self.filters:
            filter_type = f.get('Type', 'PK')
            match filter_type:
                case 'HSC':
                    b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], self.fs_base)
                case 'LSC':
                    b, a = _rbj_low_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], self.fs_base)
                case 'PK':
                    b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'], self.fs_base)
                case _:
                    raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
            coeffs.append((b, a))

            # Разделяем фильтры на НЧ и ВЧ секции
            freq = f['Frequency']
            if freq < xfade_low:
                # Чисто НЧ фильтр
                lp_coeffs.append((b, a))
            elif freq > xfade_high:
                # Чисто ВЧ фильтр
                hp_coeffs.append((b, a))
            else:
                # Фильтр в зоне кроссовера - добавляем в обе секции
                # с соответствующим ослаблением для плавного перехода
                ratio = (np.log2(freq) - np.log2(xfade_low)) / (np.log2(xfade_high) - np.log2(xfade_low))
            
                # НЧ копия с уменьшенным усилением
                lp_gain = f['Gain'] * (1 - ratio)
                if lp_gain != 0:  # Только если есть усиление
                    match filter_type:
                        case 'HSC':
                            b_lp, a_lp = _rbj_high_shelf_coeffs(freq, f['Q'], lp_gain, self.fs_base)
                        case 'LSC':
                            b_lp, a_lp = _rbj_low_shelf_coeffs(freq, f['Q'], lp_gain, self.fs_base)
                        case 'PK':
                            b_lp, a_lp = _rbj_peaking_eq_coeffs(freq, f['Q'], lp_gain, self.fs_base)
                    lp_coeffs.append((b_lp, a_lp))
                
                # ВЧ копия с уменьшенным усилением
                hp_gain = f['Gain'] * ratio
                if hp_gain != 0:  # Только если есть усиление
                    match filter_type:
                        case 'HSC':
                            b_hp, a_hp = _rbj_high_shelf_coeffs(freq, f['Q'], hp_gain, self.fs_base)
                        case 'LSC':
                            b_hp, a_hp = _rbj_low_shelf_coeffs(freq, f['Q'], hp_gain, self.fs_base)
                        case 'PK':
                            b_hp, a_hp = _rbj_peaking_eq_coeffs(freq, f['Q'], hp_gain, self.fs_base)
                    hp_coeffs.append((b_hp, a_hp))
                
        if self.subsonic is not None:
            b, a = _rbj_bessel_hpf_coeffs(self.subsonic, self.fs_base)
            coeffs.append((b, a))
            lp_coeffs.append((b, a))
            
        return coeffs, lp_coeffs, hp_coeffs
    
    def _resample(
            self,
            ir: np.ndarray, 
            fs: int,
            type = 'default',
            f_correction: int = 0
        ) -> np.ndarray:
        """
        Ресемплинг импульсной характеристики через Фурье-преобразование.
        При апсемплинге высокие частоты заполняются нулями (анти-алиасинг).
        
        Parameters
        ----------
        ir : np.ndarray
            Исходная импульсная характеристика
        fs : int
            Целевая частота дискретизации
        
        Returns
        -------
        np.ndarray
            Ресемплированная импульсная характеристика
        """
        if self.fs_base == fs:
            return ir

        # Вычисляем спектр исходной ИХ
        spectrum = transform(ir, type)
        n_original = len(spectrum)
        
        # Вычисляем новую длину, сохраняя временную длительность сигнала
        duration = n_original / self.fs_base
        n_target = int(duration * fs) + f_correction   # новое количество отсчетов

        if fs > self.fs_base:
        # Апсемплинг: экстраполируем высокие частоты
            spectrum_target = np.zeros(n_target, dtype=np.float64)
            
            # Копируем низкие частоты до fs
            nyquist_bins = n_original
            spectrum_target[:nyquist_bins] = spectrum[:nyquist_bins]
            
            # Экстраполируем высокие частоты значением на частоте fs
            last_value = spectrum[nyquist_bins-1]
            spectrum_target[nyquist_bins:n_target-1] = last_value
            
            # Преобразуем обратно во временную область
            ir_resampled = itransform(spectrum_target, type)

        else:
        # Даунсемплинг с резким обрезанием высоких частот
            ir_resampled = itransform(spectrum[:n_target], type)

        return ir_resampled

    def peak_norm(self, ir: np.ndarray) -> np.ndarray:
        """
        Нормализует импульс так, чтобы максимальный пик его АЧХ 
        соответствовал уровню self.preamp.
        Компенсирует любые потери уровня при ресемплинге.
        """
        if len(ir) == 0:
            return ir
            
        # 1. Вычисляем размер FFT. 
        # Используем 16-кратный оверсемплинг в частотной области или минимум 65536 точек.
        # Это нужно, чтобы найти "истинный" пик, который может попасть между стандартными бинами FFT.
        n_fft = max(len(ir) * 16, 65536)
        
        # 2. Получаем амплитудный спектр
        spectrum = fft(ir, n=n_fft)
        
        # 3. Находим текущий максимум магнитуды
        current_max = np.max(spectrum)
        
        # 5. Применяем усиление
        gain = self.preamp / current_max
        return ir * gain

    def min_phase(
        self,
        fs: int, 
        length: int = None, 
        fade: bool = False,
        coeffs: list = None,
        norm: bool = True,
    ) -> np.ndarray:
        """
        Создает импульсную характеристику с минимальной фазой.
        
        Parameters
        ----------
        fs : int
            Целевая частота дискретизации для выходной ИХ
        length : int, optional
            Длина выходной ИХ. По умолчанию = fs
        fade : bool, optional
            Применить затухание в конце ИХ
        coeffs : list, optional
            Коэффициенты фильтров. Если не указаны, используются кэшированные.

        Returns
        -------
        np.ndarray
            Импульсная характеристика
        """
        if length is None:
            length = fs
            
        if coeffs is None:
            coeffs = self._coeffs_cache

        # Длина для расчета на опорной частоте
        length_base = int(length * self.fs_base / fs) if self.fs_base != fs else length
        
        # Создаем импульс для опорной частоты
        out = np.zeros(length_base)
        out[0] = self.preamp
        
        # Применяем кэшированные коэффициенты фильтров
        for b, a in coeffs:
            out = lfilter(b, a, out)
            
        if self.tilt != 0.0:
            # Преобразуем импульсную характеристику в частотную область через ДКП
            X = dct(out, type=2, norm='ortho')
            
            # Создаем массив частот для ДКП
            freqs = np.linspace(0, self.fs_base/2, len(out))
            
            # Защита от деления на ноль при нулевой частоте
            freqs_safe = freqs.copy()
            freqs_safe[0] = self.pivot_hz

            # Вычисляем коэффициенты наклона для каждой частоты
            tilt_factor = db_to_amp(self.tilt * np.log2(freqs_safe / self.pivot_hz))

            # Применяем наклон к спектру
            X *= tilt_factor

            # Преобразуем обратно во временную область через обратное ДКП
            out = idct(X, type=2, norm='ortho')
     
        # Ресемплинг до целевой частоты, если необходимо
        if self.fs_base != fs:
            out = self._resample(out, fs, 'min_phase', 0) # было +32 семпла, теперь вроде не надо
            # Обрезаем или дополняем до нужной длины
            if len(out) > length:
                out = out[:length]
        if norm:
            out = self.peak_norm(out)
        
        if fade:
            # Делаем S-образное затухание в конце импульсной характеристики
            fade_len = length_base // 3
            x = np.linspace(-3, 3, fade_len)
            fade_out = 0.5 * (1 - np.tanh(x))
            out[-fade_len:] *= fade_out

        return out

    def lin_phase(
        self,
        fs: int, 
        length: int = None, 
        fade: bool = False,
        coeffs: list = None,
        norm: bool = True,
    ) -> np.ndarray:
        """
        Создает импульсную характеристику с линейной фазой.
        
        Parameters
        ----------
        fs : int
            Целевая частота дискретизации для выходной ИХ
        length : int, optional
            Длина выходной ИХ. По умолчанию = fs
        fade : bool, optional
            Применить затухание в конце ИХ
        coeffs : list, optional
            Коэффициенты фильтров. Если не указаны, используются кэшированные.

        Returns
        -------
        np.ndarray
            Импульсная характеристика с линейной фазой
        """
        if coeffs is None:
            coeffs = self._coeffs_cache

        if length is None:
            length = fs
        
        # Длина для расчета на опорной частоте
        length_base = int(length * self.fs_base / fs) if self.fs_base != fs else length
        
        w = np.linspace(0, np.pi, length_base//2 + 1)
        h_total = np.ones_like(w, np.complex128)

        # Применяем кэшированные коэффициенты фильтров
        for b, a in coeffs:
            h_total *= freqz(b, a, worN=w)[1]

        # ---------- наклон ----------
        if self.tilt != 0.0:
            # Создаем массив частот для наклона
            freqs = np.linspace(0, self.fs_base / 2, len(w))

            # Защита от деления на ноль при нулевой частоте
            freqs_safe = freqs.copy()
            freqs_safe[0] = self.pivot_hz
            
            # Вычисляем коэффициенты наклона для каждой частоты
            # ИИ выдал первую строчку для этой части кода, но в ней отсутствует подъем на НЧ
            # Если применять вторую, то при использовании в комбинированной свертке возникают артефакты и переусиление на НЧ
            # Поэтому пока в качестве костыля использует параметр fade, потому что при вызове из миксед он фолс
            tilt_factor = (db_to_amp(self.tilt * np.log2(freqs_safe / self.pivot_hz)) if fade else
                        db_to_amp(self.tilt * np.log2(np.maximum(freqs, self.pivot_hz) / self.pivot_hz)))

            # Применяем наклон к амплитудному спектру
            h_total *= tilt_factor

        # ---------- IFFT ----------
        mag = np.abs(h_total)
        out = np.fft.irfft(mag, n=length_base)
        out = np.roll(out, -length_base//2)
        
        # Ресемплинг до целевой частоты, если необходимо
        if self.fs_base != fs:
            half_fir = out[len(out) // 2:]
            half_fir = self._resample(half_fir, fs, 'lin_phase', 0)
            out = np.concatenate([np.flipud(half_fir), half_fir[1:], half_fir[-1:]], axis=None)

            # Обрезаем или дополняем до нужной длины
            if len(out) > length:
                out = out[:length]
        if norm:
            out = self.peak_norm(out)

        if fade:
            # S-образное возрастание в первой четверти
            fade_len = length // 4
            fade_in = 0.5 * (1 + np.tanh(np.linspace(-3, 3, fade_len)))
            out[:fade_len] *= fade_in

            # S-образное затухание в последней четверти
            fade_out = 0.5 * (1 - np.tanh(np.linspace(-3, 3, fade_len)))
            out[-fade_len:] *= fade_out
    
        return out

    def mix_phase(
        self,
        fs: int,
        length: int,
        fade: bool = False,
        mix_ratio: int = 16,
    ) -> np.ndarray:
        """
        Создает импульсную характеристику со смешанной фазой: 
        низ — минимальная фаза, верх — линейная фаза.
        
        Parameters
        ----------
        fs : int
            Целевая частота дискретизации для выходной ИХ
        length : int
            Длина выходной ИХ
        fade : bool, optional
            Применить затухание в конце ИХ
        mix_ratio : int, optional
            соотношение HF как 1/mix_ratio

        Returns
        -------
        np.ndarray
            Импульсная характеристика со смешанной фазой
        """
        hp_len = length // mix_ratio
        lp_len = length

        # Получаем ИХ для НЧ (минимальная фаза) и ВЧ (линейная фаза) частей
        lp_ir = self.min_phase(fs, length=lp_len, coeffs=self._lp_coeffs_cache, fade=False, norm=False)
        hp_ir = self.lin_phase(fs, length=hp_len, coeffs=self._hp_coeffs_cache, fade=False, norm=False)

        # Правильная задержка для совмещения фаз
        delay_samples = hp_len // 2
        lp_ir_delayed = np.zeros_like(lp_ir)
        lp_ir_delayed[delay_samples:] = lp_ir[:-delay_samples]
        
        # Выполняем свертку НЧ и ВЧ частей
        out = oaconvolve(lp_ir_delayed, hp_ir, mode='same')
        
        # Нормализуем энергию свернутого отклика к эталонной энергии минимальной фазы на базовой частоте
        out = self.peak_norm(out)

        if fade:
            # Делаем S-образное затухание в конце импульсной характеристики
            fade_len = len(out) // 2
            x = np.linspace(-3, 3, fade_len)
            fade_out = 0.5 * (1 - np.tanh(x))
            out[-fade_len:] *= fade_out
        
        return out

def _rbj_peaking_eq_coeffs(f0, Q, gain_db, fs):
    """
    Вычисляет коэффициенты RBJ peaking EQ biquad-фильтра.
    Возвращает (b, a) для lfilter.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


def _rbj_high_shelf_coeffs(f0, Q, gain_db, fs):
    """
    Вычисляет коэффициенты RBJ high shelf biquad-фильтра.
    Возвращает (b, a) для lfilter.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 =    A*((A+1)+(A-1)*cos_w0+2*sqrtA*alpha)
    b1 = -2*A*((A-1)+(A+1)*cos_w0)
    b2 =    A*((A+1)+(A-1)*cos_w0-2*sqrtA*alpha)
    a0 =       (A+1)-(A-1)*cos_w0+2*sqrtA*alpha
    a1 =  2*((A-1)-(A+1)*cos_w0)
    a2 =      (A+1)-(A-1)*cos_w0-2*sqrtA*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


def _rbj_low_shelf_coeffs(f0, Q, gain_db, fs):
    """
    Вычисляет коэффициенты RBJ low shelf biquad-фильтра.
    Возвращает (b, a) для lfilter.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 =    A*((A+1)-(A-1)*cos_w0+2*sqrtA*alpha)
    b1 =  2*A*((A-1)-(A+1)*cos_w0)
    b2 =    A*((A+1)-(A-1)*cos_w0-2*sqrtA*alpha)
    a0 =       (A+1)+(A-1)*cos_w0+2*sqrtA*alpha
    a1 = -2*((A-1)+(A+1)*cos_w0)
    a2 =      (A+1)+(A-1)*cos_w0-2*sqrtA*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


def _rbj_bessel_hpf_coeffs(f0: float, fs: float, order: int = 3, norm: str = "phase"):
    """
    Коэффициенты Bessel HPF произвольного порядка для lfilter.

    Parameters
    ----------
    f0 : float
        Частота среза (Гц).
    fs : float
        Частота дискретизации (Гц).
    order : int, optional
        Порядок фильтра (>=1). 1 — то же, что было раньше.  Default is 1.
    norm : {"phase", "mag", None}, optional
        Способ нормировки SciPy. По умолчанию 'phase' (максимально-плоская фазовая задержка).

    Returns
    -------
    b, a : np.ndarray
        Номератор и деноминатор для scipy.signal.lfilter (a[0] == 1).
    """
    # SciPy 1.3+ принимает f0 в Гц, если указан fs
    b, a = bessel(
        N=order,
        Wn=f0,
        btype="highpass",
        analog=False,
        output="ba",
        fs=fs,
        norm=norm,          # 'phase' по умолчанию
    )

    # Гарантируем a[0] == 1
    if not np.isclose(a[0], 1.0):
        b /= a[0]
        a /= a[0]
    return b, a


def save_ir_to_wav(ir: np.ndarray, path: str, fs: int):
    """
    Сохраняет массив ir как WAV-файл с частотой дискретизации fs в формате 32-бит float.
    """
    wavfile.write(path, fs, ir.astype(np.float32))

def db_to_amp(gain_db: float) -> float:
    """
    Преобразует усиление из дБ в амплитудный множитель.
    """
    return 10 ** (gain_db / 20)

def transform(
        ir: np.ndarray,
        type: str = 'default'
    ) -> np.ndarray:
    """
    Применяет преобразование к импульсной характеристике.
    Parameters
    ----------
    ir : np.ndarray
        Входная импульсная характеристика
        type : str, optional
        Тип преобразования: 'default', 'min_phase', 'lin_phase'
    Returns
    -------
        np.ndarray
        Преобразованная импульсная характеристика
    """
    match type:
        case 'default':
            return fft(ir)
        case 'lin_phase':
            return dct(ir, type=3, norm='forward')
        case 'min_phase':
            return dct(ir, type=3, norm='ortho')

def itransform(
        spectrum: np.ndarray,
        type: str = 'default'
    ) -> np.ndarray:
    """
    Применяет обратное преобразование к спектру импульсной характеристики.
    Parameters
    ----------
    spectrum : np.ndarray
        Входной спектр импульсной характеристики
        type : str, optional
        Тип преобразования: 'default', 'min_phase', 'lin_phase'
    Returns
    -------
        np.ndarray
        Обратное преобразованная импульсная характеристика
    """
    match type:
        case 'default':
            return ifft(spectrum).real
        case 'lin_phase':
            return idct(spectrum, type=3, norm='backward')
        case 'min_phase':
            return idct(spectrum, type=3, norm='ortho')

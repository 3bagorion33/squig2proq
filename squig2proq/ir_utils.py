""" TODO: Дописать автоматическое нормирование импульса по НЧ
    Возможно объединить функции fir_from_peq_min_phase и fir_from_peq_linear_phase
    в одну, которая принимает параметр phase='min' или 'linear'
"""

import numpy as np
from scipy.signal import lfilter, freqz, bessel, oaconvolve
from scipy.io import wavfile
from scipy.fft import dct, idct, fft, ifft


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


def _rbj_bessel_lpf_coeffs(f0, fs):
    """
    Вычисляет коэффициенты RBJ для Бесселя LPF первого порядка.
    Возвращает (b, a) для lfilter.
    """
    w0 = 2 * np.pi * f0 / fs
    K = np.tan(w0 / 2)
    norm = 1 + K
    b = np.array([K, K]) / norm
    a = np.array([1, (K - 1) / norm])
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


def _fourier_resample(
        ir: np.ndarray, 
        fs_original: int, 
        fs_target: int,
        f_correction: int = 0
    ) -> np.ndarray:
    """
    Ресемплинг импульсной характеристики через Фурье-преобразование.
    При апсемплинге высокие частоты заполняются нулями (анти-алиасинг).
    
    Parameters
    ----------
    ir : np.ndarray
        Исходная импульсная характеристика
    fs_original : int
        Исходная частота дискретизации
    fs_target : int
        Целевая частота дискретизации
    
    Returns
    -------
    np.ndarray
        Ресемплированная импульсная характеристика
    """
    if fs_original == fs_target:
        return ir
    
    # Вычисляем спектр исходной ИХ
    spectrum = fft(ir)
    n_original = len(spectrum)
    
    # Вычисляем новую длину, сохраняя временную длительность сигнала
    duration = n_original / fs_original
    n_target = int(duration * fs_target) + f_correction   # новое количество отсчетов
    
    if fs_target > fs_original:
       # Апсемплинг: экстраполируем высокие частоты
        spectrum_target = np.zeros(n_target, dtype=complex)
        
        # Копируем низкие частоты (до частоты Найквиста исходного сигнала)
        nyquist_bins = n_original // 2 + 1
        spectrum_target[:nyquist_bins] = spectrum[:nyquist_bins]
        
        # Экстраполируем высокие частоты значением на частоте Найквиста
        # Для положительных частот
        if nyquist_bins < n_target // 2 + 1:
            last_value = spectrum[nyquist_bins - 1]  # Последнее значение на Найквисте
            spectrum_target[nyquist_bins:n_target//2 + 1] = last_value
        
        # Восстанавливаем эрмитову симметрию для отрицательных частот
        if n_original % 2 == 0:  # четная длина
            # Копируем симметричную часть исходного спектра
            spectrum_target[n_target-nyquist_bins+2:] = np.conj(spectrum[nyquist_bins-2:0:-1])
            # Экстраполируем отрицательные высокие частоты
            if n_target // 2 + 1 > nyquist_bins:
                last_negative = np.conj(last_value)
                spectrum_target[n_target//2 + 1:n_target-nyquist_bins+2] = last_negative
        else:  # нечетная длина
            spectrum_target[n_target-nyquist_bins+1:] = np.conj(spectrum[nyquist_bins-1:0:-1])
            if n_target // 2 + 1 > nyquist_bins:
                last_negative = np.conj(last_value)
                spectrum_target[n_target//2 + 1:n_target-nyquist_bins+1] = last_negative
    else:
        # Даунсемплинг: обрезаем высокие частоты
        spectrum_target = np.zeros(n_target, dtype=complex)
        
        # Копируем низкие частоты до новой частоты Найквиста
        nyquist_bins_target = n_target // 2 + 1
        spectrum_target[:nyquist_bins_target] = spectrum[:nyquist_bins_target]
        
        # Восстанавливаем эрмитову симметрию для отрицательных частот
        if n_target % 2 == 0:  # четная длина
            spectrum_target[n_target-nyquist_bins_target+2:] = np.conj(spectrum_target[nyquist_bins_target-2:0:-1])
        else:  # нечетная длина
            spectrum_target[n_target-nyquist_bins_target+1:] = np.conj(spectrum_target[nyquist_bins_target-1:0:-1])

    # Преобразуем обратно во временную область
    ir_resampled = ifft(spectrum_target).real
    return ir_resampled


def fir_from_peq_min_phase(
        filters: list, 
        fs: int, 
        length: int = None, 
        preamp: float = 0.0, 
        tilt: float = 0.0, 
        pivot_hz: float = 1000.0, 
        fade: bool = False,
        subsonic: float = None,
        fs_base: int = 48000,  # Опорная частота для расчета коэффициентов
        attenuation: float = 1.0,
    ) -> np.ndarray:
    """
    Строит цепочку IIR-фильтров (RBJ biquad) и возвращает импульсную характеристику.
    Коэффициенты рассчитываются для опорной частоты fs_base, затем выполняется ресемплинг до fs.
    
    Parameters
    ----------
    filters : list
        Список параметрических фильтров
    fs : int
        Целевая частота дискретизации для выходной ИХ
    length : int, optional
        Длина выходной ИХ. По умолчанию = fs
    preamp : float, optional
        Предусиление в дБ
    tilt : float, optional
        Наклон АЧХ в дБ/октаву
    pivot_hz : float, optional
        Опорная частота для наклона
    fade : bool, optional
        Применить затухание в конце ИХ
    subsonic : float, optional
        Частота среза ВЧ фильтра
    fs_base : int, optional
        Опорная частота для расчета коэффициентов. По умолчанию = fs
    attenuation : float, optional
        Аттенюация коэффициентов

    Returns
    -------
    np.ndarray
        Импульсная характеристика
    """
    if fs_base is None:
        fs_base = fs
    
    if length is None:
        length = fs
        
    # Длина для расчета на опорной частоте
    length_base = int(length * fs_base / fs) if fs_base != fs else length
    
    # Создаем импульс для опорной частоты
    out = np.zeros(length_base)
    out[0] = 10 ** (preamp / 20)
    
    # Применяем фильтры, используя опорную частоту fs_base
    for f in filters:
        filter_type = f.get('Type', 'PK')
        match filter_type:
            case 'HSC':
                b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'] / attenuation, fs_base)
            case 'LSC':
                b, a = _rbj_low_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'] / attenuation, fs_base)
            case 'PK':
                b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'] / attenuation, fs_base)
            case _:
                raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
        out = lfilter(b, a, out)
        
    if subsonic is not None:
        b, a = _rbj_bessel_hpf_coeffs(subsonic, fs_base)
        out = lfilter(b, a, out)
        
    if tilt != 0.0:
        # Преобразуем импульсную характеристику в частотную область через ДКП
        X = dct(out, type=2, norm='ortho')
        
        # Создаем массив частот для ДКП (линейно от 0 до fs_base/2)
        freqs = np.linspace(0, fs_base/2, len(out))
        
        # Защита от деления на ноль при нулевой частоте
        freqs_safe = freqs.copy()
        freqs_safe[0] = pivot_hz
        
        # Вычисляем коэффициенты наклона для каждой частоты
        tilt_factor = 10 ** (tilt * np.log2(freqs_safe / pivot_hz) / 20)
        
        # Применяем наклон к спектру
        X *= tilt_factor
        
        # Преобразуем обратно во времную область через обратное ДКП
        out = idct(X, type=2, norm='ortho')
 
    # Ресемплинг до целевой частоты, если необходимо
    if fs_base != fs:
        out = _fourier_resample(out, fs_base, fs, 32)
        out *= 10 ** (0.18 / 20)
        # Обрезаем или дополняем до нужной длины
        if len(out) > length:
            out = out[:length]
    
    if fade:
        # Делаем S-образное затухание в конце импульсной характеристики
        fade_len = length // 3
        x = np.linspace(-3, 3, fade_len)  # -3..3 — диапазон для плавности
        fade_out = 0.5 * (1 - np.tanh(x))  # S-образная кривая от 1 до 0
        out[-fade_len:] *= fade_out

    return out


def fir_from_peq_linear_phase(
        filters: list, 
        fs: int, 
        length: int = None, 
        preamp: float = 0.0, 
        tilt: float = 0.0, 
        pivot_hz: float = 1000.0,
        fade: bool = False,
        subsonic: float = None,
        fs_base: int = 48000,  # Опорная частота для расчета коэффициентов
    ) -> np.ndarray:
    """
    Строит целевую АЧХ по фильтрам и вычисляет линейно-фазовый FIR-фильтр через БПФ/обратный БПФ.
    Коэффициенты рассчитываются для опорной частоты fs_base, затем выполняется ресемплинг до fs.
    
    Parameters
    ----------
    filters : list
        Список параметрических фильтров
    fs : int
        Целевая частота дискретизации для выходной ИХ
    length : int, optional
        Длина выходной ИХ. По умолчанию = fs
    preamp : float, optional
        Предусиление в дБ
    tilt : float, optional
        Наклон АЧХ в дБ/октаву
    pivot_hz : float, optional
        Опорная частота для наклона
    fade : bool, optional
        Применить затухание в конце ИХ
    subsonic : float, optional
        Частота среза ВЧ фильтра
    fs_base : int, optional
        Опорная частота для расчета коэффициентов. По умолчанию = fs
        
    Returns
    -------
    np.ndarray
        Импульсная характеристика
    """
    if fs_base is None:
        fs_base = fs
        
    if length is None:
        length = fs
    
    # Длина для расчета на опорной частоте
    length_base = int(length * fs_base / fs) if fs_base != fs else length
    
    w = np.linspace(0, np.pi, length_base//2 + 1)
    h_total = (10 ** ((preamp - 2.196) / 20)) * np.ones_like(w, np.complex128)

    # ---------- складываем PEQ ----------
    for f in filters:
        filter_type = f.get('Type', 'PK')
        match filter_type:
            case 'HSC':
                b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], fs_base)
            case 'LSC':
                b, a = _rbj_low_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], fs_base)
            case 'PK':
                b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'], fs_base)
            case _:
                raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
        h_total *= freqz(b, a, worN=w)[1]
        
    if subsonic is not None:
        b, a = _rbj_bessel_hpf_coeffs(subsonic, fs_base)
        h_total *= freqz(b, a, worN=w)[1]

    # ---------- наклон ----------
    if tilt != 0.0:
        # Создаем массив частот для наклона
        freqs = np.linspace(0, fs_base / 2, len(w))

        # Защита от деления на ноль при нулевой частоте
        freqs_safe = freqs.copy()
        freqs_safe[0] = pivot_hz
        
        # Вычисляем коэффициенты наклона для каждой частоты
        # ИИ выдал первую строчку для этой части кода, но в ней отсутствует подъем на НЧ
        # Если применять вторую, то при использовании в комбинированной свертке возникают артефакты и переусиление на НЧ
        # Поэтому пока в качестве костыля использует параметр fade, потому что при вызове из миксед он фолс
        tilt_factor = (10 ** (tilt * np.log2(freqs_safe / pivot_hz) / 20) if fade else
                       10 ** (tilt * np.log2(np.maximum(freqs, pivot_hz) / pivot_hz) / 20))

        # Применяем наклон к амплитудному спектру
        h_total *= tilt_factor

    # ---------- IFFT ----------
    mag = np.abs(h_total)
    fir = np.fft.irfft(mag, n=length_base)
    fir = np.roll(fir, -length_base//2)
    
    # Ресемплинг до целевой частоты, если необходимо
    if fs_base != fs:
        half_fir = fir[len(fir) // 2:]
        half_fir = _fourier_resample(half_fir, fs_base, fs, 0)
        fir = np.concatenate([np.flipud(half_fir), half_fir[1:], half_fir[-1:]], axis=None)

        fir *= 10 ** (-0.08 / 20)
        # Обрезаем или дополняем до нужной длины
        if len(fir) > length:
            fir = fir[:length]

    if fade:
        # S-образное возрастание в первой четверти
        fade_len = length // 4
        fade_in = 0.5 * (1 + np.tanh(np.linspace(-3, 3, fade_len)))
        fir[:fade_len] *= fade_in

        # S-образное затухание в последней четверти
        fade_out = 0.5 * (1 - np.tanh(np.linspace(-3, 3, fade_len)))
        fir[-fade_len:] *= fade_out
 
    return fir


def fir_from_peq_mixed_phase(
        filters: list,
        fs: int,
        length: int,
        crossover: float = 3000.0,          # частота сшивки, Гц
        xfade_width: float = 0.2,           # ширина кроссовера (в октавах)
        preamp: float = 0.0,
        tilt: float = 0.0,
        pivot_hz: float = 3000.0,
        fade: bool = False,
        subsonic: float = None,
        fs_base: int = 48000,  # Опорная частота для расчета коэффициентов
    ) -> np.ndarray:
    """
    Строит «смешанный» FIR: низ — минимальная фаза, верх — линейная фаза.
    Коэффициенты рассчитываются для опорной частоты fs_base, затем выполняется ресемплинг до fs.
    
    Parameters
    ----------
    filters : list
        Список параметрических фильтров
    fs : int
        Целевая частота дискретизации для выходной ИХ
    length : int
        Длина выходной ИХ
    crossover : float, optional
        Частота сшивки (Гц)
    xfade_width : float, optional
        Ширина кроссовера (в октавах)
    preamp : float, optional
        Общий предусилитель (дБ)
    tilt : float, optional
        Наклон АЧХ в дБ/октаву
    pivot_hz : float, optional
        Опорная частота для наклона
    fade : bool, optional
        Применить затухание в конце ИХ
    subsonic : float, optional
        Частота среза ВЧ фильтра
    fs_base : int, optional
        Опорная частота для расчета коэффициентов. По умолчанию = fs
        
    Returns
    -------
    np.ndarray
        Импульсная характеристика
    """
    if fs_base is None:
        fs_base = fs
    
    hp_len = length // 16
    lp_len = length

    # Разделяем фильтры на НЧ и ВЧ секции
    lp_filters = []
    hp_filters = []
    xfade_low = crossover * 2**(-xfade_width/2)
    xfade_high = crossover * 2**(xfade_width/2)
    
    for f in filters:
        freq = f['Frequency']
        # Копируем фильтр, чтобы не модифицировать оригинал
        f_copy = f.copy()
        
        if freq < xfade_low:
            # Чисто НЧ фильтр
            lp_filters.append(f_copy)
        elif freq > xfade_high:
            # Чисто ВЧ фильтр
            hp_filters.append(f_copy)
        else:
            # Фильтр в зоне кроссовера - добавляем в обе секции
            # с соответствующим ослаблением для плавного перехода
            ratio = (np.log2(freq) - np.log2(xfade_low)) / (np.log2(xfade_high) - np.log2(xfade_low))
            
            # НЧ копия с уменьшенным усилением
            lp_copy = f_copy.copy()
            lp_copy['Gain'] *= (1 - ratio)
            lp_filters.append(lp_copy)
            
            # ВЧ копия с уменьшенным усилением
            hp_copy = f_copy.copy()
            hp_copy['Gain'] *= ratio
            hp_filters.append(hp_copy)
    
    # Получаем ИХ для НЧ (минимальная фаза) и ВЧ (линейная фаза) частей
    lp_ir = fir_from_peq_min_phase(lp_filters, fs, length=lp_len, tilt=tilt, pivot_hz=pivot_hz, subsonic=subsonic, fs_base=fs_base)
    hp_ir = fir_from_peq_linear_phase(hp_filters, fs, length=hp_len, tilt=tilt, pivot_hz=pivot_hz, fs_base=fs_base, fade=True)

    # Правильная задержка для совмещения фаз
    delay_samples = hp_len // 2
    lp_ir_delayed = np.zeros_like(lp_ir)
    lp_ir_delayed[delay_samples:] = lp_ir[:-delay_samples]
    
    # Выполняем свертку НЧ и ВЧ частей
    out = oaconvolve(lp_ir_delayed, hp_ir, mode='same')
    
    # Применяем предусиление
    out *= 10 ** ((preamp + 1.55) / 20)

    # Ресемплинг до целевой частоты, если необходимо
    if fs_base != fs:
        # Точно подгоняем под нужную длину
        out *= 10 ** (-0.035 / 20)
        if len(out) > length:
            out = out[:length]

    if fade:
        # Делаем S-образное затухание в конце импульсной характеристики
        fade_len = len(out) // 2
        x = np.linspace(-3, 3, fade_len)  # -3..3 — диапазон для плавности
        fade_out = 0.5 * (1 - np.tanh(x))  # S-образная кривая от 1 до 0
        out[-fade_len:] *= fade_out
    
    return out


def save_ir_to_wav(ir: np.ndarray, path: str, fs: int):
    """
    Сохраняет массив ir как WAV-файл с частотой дискретизации fs в формате 32-бит float.
    """
    wavfile.write(path, fs, ir.astype(np.float32))

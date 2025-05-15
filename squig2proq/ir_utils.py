import numpy as np
from scipy.signal import lfilter, freqz, bessel
from scipy.io import wavfile
from scipy.fft import dct, idct 


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


def fir_from_peq_min_phase(
        filters: list, fs: int, 
        length: int = None, preamp: float = 0.0, 
        tilt: float = 0.0, 
        pivot_hz: float = 1000.0, 
        fade: bool = False,
        subsonic: float = None,          
    ) -> np.ndarray:
    """
    Строит цепочку IIR-фильтров (RBJ biquad) и возвращает импульсную характеристику.
    Длина фильтра по умолчанию = fs.
    preamp — предусиление в дБ, применяется к импульсу.
    tilt — наклон АЧХ в дБ/октаву
    """
    if length is None:
        length = fs
    impulse = np.zeros(length)
    impulse[0] = 10 ** (preamp / 20)
    out = impulse.copy()
    for f in filters:
        if f.get('Type', 'PK') == 'HSC':
            b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
        else:
            b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
        out = lfilter(b, a, out)
    if subsonic is not None:
        b, a = _rbj_bessel_hpf_coeffs(subsonic, fs)
        out = lfilter(b, a, out)
    if tilt != 0.0:
        # Преобразуем импульсную характеристику в частотную область через ДКП
        X = dct(out, type=2, norm='ortho')
        
        # Создаем массив частот для ДКП (линейно от 0 до fs/2)
        freqs = np.linspace(0, fs/2, len(out))
        
        # Защита от деления на ноль при нулевой частоте
        freqs_safe = freqs.copy()
        freqs_safe[0] = pivot_hz
        
        # Вычисляем коэффициенты наклона для каждой частоты
        tilt_factor = 10 ** (tilt * np.log2(freqs_safe / pivot_hz) / 20)
        
        # Применяем наклон к спектру
        X *= tilt_factor
        
        # Преобразуем обратно во временную область через обратное ДКП
        out = idct(X, type=2, norm='ortho')
 
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
    ) -> np.ndarray:
    """
    Строит целевую АЧХ по фильтрам и вычисляет линейно-фазовый FIR-фильтр через БПФ/обратный БПФ.
    Если length не задан, длина определяется по энергетическому критерию (energy_threshold) — симметрично от центра.
    min_length — минимальная длина фильтра.
    preamp — предусиление в дБ, применяется к АЧХ.
    tilt — наклон АЧХ в дБ/октаву
    """
    if length is None:
        length = fs
    
    w   = np.linspace(0, np.pi, length//2 + 1)
    h_total = (10 ** (preamp / 20)) * np.ones_like(w, np.complex128)

    # ---------- складываем PEQ ----------
    for f in filters:
        b, a = (_rbj_high_shelf_coeffs if f.get('Type','PK')=='HSC'
                else _rbj_peaking_eq_coeffs)(f['Frequency'], f['Q'], f['Gain'], fs)
        h_total *= freqz(b, a, worN=w)[1]
    if subsonic is not None:
        b, a = _rbj_bessel_hpf_coeffs(subsonic, fs)
        h_total *= freqz(b, a, worN=w)[1]

    # ---------- наклон ----------
    if tilt != 0.0:
        # Создаем массив частот для наклона
        freqs = np.linspace(0, fs / 2, len(w))

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
    mag_full = np.concatenate([mag, mag[-2:0:-1]])
    fir = np.fft.irfft(mag_full, n=length)
    fir = np.roll(fir, -length//2)

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
    ) -> np.ndarray:
    """
    Строит «смешанный» FIR: низ — минимальная фаза, верх — линейная фаза.
    crossover      — частота сшивки (Гц)
    xfade_width    — какой диапазон вокруг crossover уходит в кросс-fade (в октавах)
    preamp         — общий предусилитель (дБ)
    tilt          — наклон АЧХ в дБ/октаву
    Остальные параметры совпадают с существующими функциями.
    """
    hp_len = length // 16
    lp_len = length # - hp_len // 2

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
    
    ###
    # Тут проблема в том, что я фильтрую фильтры до преобразования в коэффициенты. То есть делаю коррекцию параметрических в тех местах,
    # где они уже есть. Поэтому нельзя выбрать точно позицию.
    # Нужно добавить сшивку по Баттерворту первого порядка уже готовых импульсов.
    # Или, что еще лучше, фильтровать коэффециенты на момент генерации импульсов. 
    ###
    # Получаем ИХ для НЧ (минимальная фаза) и ВЧ (линейная фаза) частей
    lp_ir = fir_from_peq_min_phase(lp_filters, fs, length=lp_len, tilt=tilt, pivot_hz=pivot_hz, subsonic=subsonic)
    hp_ir = fir_from_peq_linear_phase(hp_filters, fs, length=hp_len, tilt=tilt, pivot_hz=pivot_hz)
    # Подвинуть начало импульса lp_ir на половину длины импульса hp_ir
    lp_ir = np.roll(lp_ir, hp_len // 2)
    
    # Выполняем свертку НЧ и ВЧ частей
    # Используем 'full' режим свертки, чтобы не потерять информацию
    out = np.convolve(lp_ir, hp_ir, mode='same')
    #out = out[:lp_len + hp_len // 2]  # Обрезаем до нужной длины
    
    # Нормализуем энергию относительно исходных импульсов
    #target_energy = np.sqrt(np.sum(lp_ir**2) * np.sum(hp_ir**2))
    #current_energy = np.sqrt(np.sum(out**2))
    #if current_energy > 0:  # Защита от деления на ноль
    #    out *= target_energy / current_energy
    out *= 10 ** (preamp / 20)  # Применяем предусиление
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

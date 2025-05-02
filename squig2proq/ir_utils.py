import numpy as np
from scipy.signal import lfilter, freqz
from scipy.io import wavfile


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


def impulse_response_from_peq(filters: list, fs: int, length: int = None, preamp: float = 0.0, tilt: float = 0.0) -> np.ndarray:
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
    
    if tilt != 0.0:
        out = apply_tilt_conv_min_phase(out, fs, tilt)
    
    return out


def fir_from_peq_linear_phase(filters: list, fs: int, length: int = None, energy_threshold: float = 0.999999, min_length: int = 5, preamp: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """
    Строит целевую АЧХ по фильтрам и вычисляет линейно-фазовый FIR-фильтр через БПФ/обратный БПФ.
    Если length не задан, длина определяется по энергетическому критерию (energy_threshold) — симметрично от центра.
    min_length — минимальная длина фильтра.
    preamp — предусиление в дБ, применяется к АЧХ.
    tilt — наклон АЧХ в дБ/октаву
    """
    if length is None:
        max_len = fs * 8  # 8 секунд, обычно достаточно
        n_fft = max_len
        w = np.linspace(0, np.pi, n_fft//2+1)
        h_total = np.ones_like(w, dtype=np.complex128) * (10 ** (preamp / 20))
        for f in filters:
            if f.get('Type', 'PK') == 'HSC':
                b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
            else:
                b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
            w_f, h = freqz(b, a, worN=w)
            h_total *= h
        mag = np.abs(h_total)
        mag_full = np.concatenate([mag, mag[-2:0:-1]])
        fir = np.fft.irfft(mag_full, n=n_fft)
        fir = np.roll(fir, -max_len//2)
        # Энергетический критерий (симметрично от центра)
        total_energy = np.sum(fir ** 2)
        center = max_len // 2
        win_len = min_length if min_length % 2 == 1 else min_length + 1
        while win_len < max_len:
            half = win_len // 2
            start = center - half
            end = center + half + 1
            energy = np.sum(fir[start:end] ** 2)
            if energy / total_energy >= energy_threshold:
                break
            win_len += 2  # увеличиваем окно на 2 для симметрии
        # Дополнительная проверка по -180 dBFS
        dBFS_limit = 10 ** (-180 / 20)
        fir_win = fir[start:end]
        max_val = np.max(np.abs(fir_win))
        while (np.abs(fir[start]) > max_val * dBFS_limit or np.abs(fir[end-1]) > max_val * dBFS_limit) and (start > 0 or end < len(fir)):
            if start > 0:
                start -= 1
            if end < len(fir):
                end += 1
            fir_win = fir[start:end]
            max_val = np.max(np.abs(fir_win))
        fir = fir[start:end]
        #fir *= np.hanning(len(fir))  # Применяем окно Ханнинга для сглаживания хвостов
        if tilt != 0.0:
            fir = apply_tilt_conv_linear_phase(fir, fs, tilt)
        return fir
    else:
        n_fft = length
        w = np.linspace(0, np.pi, n_fft//2+1)
        h_total = np.ones_like(w, dtype=np.complex128) * (10 ** (preamp / 20))
        for f in filters:
            if f.get('Type', 'PK') == 'HSC':
                b, a = _rbj_high_shelf_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
            else:
                b, a = _rbj_peaking_eq_coeffs(f['Frequency'], f['Q'], f['Gain'], fs)
            w_f, h = freqz(b, a, worN=w)
            h_total *= h
        mag = np.abs(h_total)
        mag_full = np.concatenate([mag, mag[-2:0:-1]])
        fir = np.fft.irfft(mag_full, n=n_fft)
        fir = np.roll(fir, -length//2)
        if tilt != 0.0:
            fir = apply_tilt_conv_linear_phase(fir, fs, tilt)
        return fir[:length]


def fir_from_peq_mixed_phase(
        filters: list,
        fs: int,
        crossover: float = 1500.0,          # частота сшивки, Гц
        lp_len: int | None = None,          # длина ИХО низкочастотной части
        hp_len: int | None = None,          # длина ИХО высокочастотной части
        xfade_width: float = 0.5,           # ширина кроссовера (в октавах)
        preamp: float = 0.0,
        energy_threshold: float = 0.999999,
        tilt: float = 0.0
    ) -> np.ndarray:
    """
    Строит «смешанный» FIR: низ — минимальная фаза, верх — линейная фаза.
    crossover      — частота сшивки (Гц)
    xfade_width    — какой диапазон вокруг crossover уходит в кросс-fade (в октавах)
    preamp         — общий предусилитель (дБ)
    tilt          — наклон АЧХ в дБ/октаву
    Остальные параметры совпадают с существующими функциями.
    """
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
    lp_ir = impulse_response_from_peq(lp_filters, fs, length=lp_len, preamp=preamp + 3.6)
    hp_ir = fir_from_peq_linear_phase(hp_filters, fs, length=hp_len, energy_threshold=energy_threshold, preamp=preamp + 3.6)
      # Выполняем свертку НЧ и ВЧ частей
    # Используем 'full' режим свертки, чтобы не потерять информацию
    mixed_ir = np.convolve(lp_ir, hp_ir, mode='full')
    
    if tilt != 0.0:
        mixed_ir = apply_tilt(mixed_ir, fs, tilt)
    
    # Нормализуем энергию относительно исходных импульсов
    target_energy = np.sqrt(np.sum(lp_ir**2) * np.sum(hp_ir**2))
    current_energy = np.sqrt(np.sum(mixed_ir**2))
    if current_energy > 0:  # Защита от деления на ноль
        mixed_ir *= target_energy / current_energy
    
    return mixed_ir


def save_ir_to_wav(ir: np.ndarray, path: str, fs: int):
    """
    Сохраняет массив ir как WAV-файл с частотой дискретизации fs в формате 32-бит float.
    """
    wavfile.write(path, fs, ir.astype(np.float32))


def apply_tilt(ir: np.ndarray, fs: int, tilt_db_per_octave: float) -> np.ndarray:
    """
    Применяет наклон (tilt) в dB/октаву к импульсной характеристике (FIR/IIR).
    tilt_db_per_octave: положительное значение — подъем ВЧ, отрицательное — подъем НЧ.
    1 dB/octave ≈ 10 dB разницы между 20 Гц и 20 кГц.
    """
    if tilt_db_per_octave == 0.0:
        return ir
    
    n = len(ir)
    # Получаем спектр
    spectrum = np.fft.rfft(ir)
    # Частоты для rfft
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    # Создаем плавный переход для очень низких частот
    transition_freq = 20.0  # Гц
    # Используем плавную сигмоиду для перехода
    transition = 1 / (1 + np.exp(-(freqs - transition_freq/4) / (transition_freq/8)))
    
    # Вычисляем наклон с защитой от log(0)
    log2f = np.zeros_like(freqs)
    nonzero_mask = freqs > 0
    log2f[nonzero_mask] = np.log2(freqs[nonzero_mask] / transition_freq)
    
    # Применяем наклон
    gain = 10 ** (tilt_db_per_octave / 20 * log2f)
    
    # Плавно переходим к единичному усилению на очень низких частотах
    gain = gain * transition + 1.0 * (1 - transition)
    
    # Сохраняем DC компоненту
    gain[0] = 1.0
    
    # Применяем наклон с сохранением фазы
    phase = np.angle(spectrum)
    mag = np.abs(spectrum) * gain
    spectrum_tilted = mag * np.exp(1j * phase)
    
    # Обратно во временную область
    ir_tilted = np.fft.irfft(spectrum_tilted, n=n)
    return ir_tilted


def apply_tilt_conv_min_phase(ir: np.ndarray, fs: int, tilt_db_per_octave: float) -> np.ndarray:
    """
    Применяет наклон (tilt) в dB/октаву к импульсной характеристике через свертку.
    Версия для фильтров с минимальной фазой.
    tilt_db_per_octave: положительное значение — подъем ВЧ, отрицательное — подъем НЧ.
    """
    if tilt_db_per_octave == 0.0:
        return ir

    # Создаем фильтр первого порядка
    fc = 1000.0  # Центральная частота около 1 кГц
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / np.sqrt(2)  # Q = 1/sqrt(2) для естественного отклика
    
    if tilt_db_per_octave > 0:
        # ФВЧ для подъема ВЧ
        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = (1 + np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    else:
        # ФНЧ для подъема НЧ
        b0 = (1 - np.cos(w0)) / 2
        b1 = 1 - np.cos(w0)
        b2 = (1 - np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0])
    
    # Вычисляем количество каскадов для достижения нужного наклона
    # 6 дБ/окт на каскад для фильтров первого порядка
    cascades = abs(tilt_db_per_octave) / 6.0
    full_cascades = int(cascades)
    remainder = cascades - full_cascades
    
    # Применяем полные каскады
    result = ir.copy()
    for _ in range(full_cascades):
        result = lfilter(b, a, result)
    
    # Применяем дробный каскад через линейную интерполяцию
    if remainder > 0:
        partial = lfilter(b, a, result)
        result = (1 - remainder) * result + remainder * partial
    
    # Нормализуем энергию
    result *= np.sqrt(np.sum(ir**2) / np.sum(result**2))
    
    return result


def apply_tilt_conv_linear_phase(ir: np.ndarray, fs: int, tilt_db_per_octave: float) -> np.ndarray:
    """
    Применяет наклон (tilt) в dB/октаву к импульсной характеристике через свертку.
    Версия для линейно-фазовых фильтров.
    tilt_db_per_octave: положительное значение — подъем ВЧ, отрицательное — подъем НЧ.
    """
    if tilt_db_per_octave == 0.0:
        return ir

    # Создаем симметричное ядро свертки для линейной фазы
    fc = 1000.0  # Центральная частота около 1 кГц
    kernel_length = min(len(ir) // 4, fs // 50)  # Ограничиваем длину ядра
    if kernel_length % 2 == 0:
        kernel_length += 1  # Делаем длину нечетной для симметрии
    
    t = np.arange(-(kernel_length // 2), kernel_length // 2 + 1) / fs
    if tilt_db_per_octave > 0:
        # ФВЧ для подъема ВЧ
        kernel = -(np.sin(2 * np.pi * fc * t) / (np.pi * t))
    else:
        # ФНЧ для подъема НЧ
        kernel = np.sin(2 * np.pi * fc * t) / (np.pi * t)
    
    # Исправляем деление на ноль в центре
    kernel[kernel_length // 2] = 2 * fc if tilt_db_per_octave > 0 else 1.0
    
    # Применяем окно Блэкмана для уменьшения звона
    kernel *= np.blackman(kernel_length)
    
    # Нормализуем ядро
    kernel /= np.sum(np.abs(kernel))
    
    # Вычисляем количество каскадов
    cascades = abs(tilt_db_per_octave) / 6.0
    full_cascades = int(cascades)
    remainder = cascades - full_cascades
    
    # Применяем полные каскады
    result = ir.copy()
    for _ in range(full_cascades):
        result = np.convolve(result, kernel, mode='same')
    
    # Применяем дробный каскад через линейную интерполяцию
    if remainder > 0:
        partial = np.convolve(result, kernel, mode='same')
        result = (1 - remainder) * result + remainder * partial
    
    # Нормализуем энергию
    result *= np.sqrt(np.sum(ir**2) / np.sum(result**2))
    
    return result

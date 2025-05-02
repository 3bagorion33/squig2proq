import re
import math

def parse_filter_text(content):
    filters = []
    preamp = 0.0
    for line in content.splitlines():
        preamp_match = re.match(r'Preamp: ([\-\d.]+) dB', line)
        if preamp_match:
            preamp = float(preamp_match.group(1))
        match = re.match(r'Filter\s?(\d*): ON (\w+) Fc (\d+) Hz Gain ([\-\d.]+) dB Q ([\d.]+)', line)
        if match:
            filters.append({
                'Type': match.group(2),
                'Frequency': int(match.group(3)),
                'Gain': float(match.group(4)),
                'Q': float(match.group(5)),
            })
    return filters, preamp

def build_ffp(filters, template_text, adjust_q=False, preamp=0.0):
    template_lines = template_text.splitlines(keepends=True)
    updated_lines = []
    band_map = {i + 1: f for i, f in enumerate(filters)}
    current_band = None

    for line in template_lines:
        if line.strip().startswith('Output Level='):
            updated_lines.append(f'Output Level={preamp / 36}\n')
            continue
        band_match = re.match(r'Band (\d+) ', line)
        if band_match:
            current_band = int(band_match.group(1))

        if current_band and current_band in band_map:
            f = band_map[current_band]
            if f"Band {current_band} Frequency=" in line:
                updated_lines.append(f"Band {current_band} Frequency={round(math.log2(f['Frequency']), 14)}\n")
            elif f"Band {current_band} Gain=" in line:
                updated_lines.append(f"Band {current_band} Gain={f['Gain']}\n")
            elif f"Band {current_band} Q=" in line:
                Q = 0.5 + 0.1355 * math.log(f['Q'] * math.sqrt(2)) if adjust_q else 0.5 + 0.1355 * math.log(f['Q'])
                updated_lines.append(f"Band {current_band} Q={round(Q, 14)}\n")
            elif f"Band {current_band} Used=" in line:
                updated_lines.append(f"Band {current_band} Used=1\n")
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    return updated_lines

def truncate_middle(path, max_length=50):
    if len(path) <= max_length:
        return path
    part_length = (max_length - 3) // 2
    return path[:part_length] + '...' + path[-part_length:]

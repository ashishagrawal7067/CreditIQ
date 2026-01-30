import re

with open('app.py', encoding='utf-8') as f:
    content = f.read()

def rgba_to_tuple(m):
    vals = [v.strip() for v in m.group(1).split(',')]
    r = float(vals[0]) / 255
    g = float(vals[1]) / 255
    b = float(vals[2]) / 255
    a = float(vals[3])
    return f'({r:.3f},{g:.3f},{b:.3f},{a})'

fixed = re.sub(r"'rgba\(([^)]+)\)'", rgba_to_tuple, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Done — all rgba() values converted to matplotlib tuples')
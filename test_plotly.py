import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Dados
clientes = ['Cliente A', 'Cliente B', 'Cliente C', 'Cliente D', 'Cliente E']
valores = [1500, 1200, 950, 800, 650]

# Figura
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0b0f14')
ax.set_facecolor('#0b0f14')

y_pos = np.arange(len(clientes))
altura = 0.5
cor_barra = '#ff9500'

# Offset para evitar clipping
x_offset = 2
rounding = 12

def barra_arredondada(ax, x, y, width, height):
    bar = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=0,
        facecolor=cor_barra
    )
    ax.add_patch(bar)

# Plot
for i, valor in enumerate(valores):
    barra_arredondada(ax, x_offset, y_pos[i], valor - x_offset, altura)
    ax.text(valor + 20, y_pos[i], f'{valor}k',
            va='center', ha='left', color='white', fontsize=10)

# Eixos
ax.set_yticks(y_pos)
ax.set_yticklabels(clientes, color='white')
ax.invert_yaxis()
ax.set_xlim(0, 1600)

ax.tick_params(axis='x', colors='white')
ax.grid(axis='x', alpha=0.08)
ax.set_title('Vendas por Cliente (Top 5)', color='white', fontsize=14)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()

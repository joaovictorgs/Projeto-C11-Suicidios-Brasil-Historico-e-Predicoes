import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('images', exist_ok=True)

df = pd.read_csv('data/brazil_aggregated.csv')

plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['No of Suicides'], marker='o', linewidth=2, markersize=4, color='#e74c3c')
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Número de Suicídios', fontsize=12)
plt.title('Série Temporal de Suicídios no Brasil (1979-2020)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/time_series_suicidios_brasil.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo em: images/time_series_suicidios_brasil.png")

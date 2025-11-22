import pandas as pd
import matplotlib.pyplot as plt
import os

# Criar diretório de imagens se não existir
os.makedirs('images', exist_ok=True)

# Carregar dados globais
df = pd.read_csv('data/combined_processed_data.csv')

# Agregar o número total de suicídios por ano (somando todos os países)
df_global = df.groupby('Year')['No of Suicides'].sum().reset_index()

# Plotar série temporal global
plt.figure(figsize=(14, 7))
plt.plot(df_global['Year'], df_global['No of Suicides'], marker='o', linewidth=2, markersize=5, color='#e74c3c')
plt.xlabel('Ano', fontsize=13)
plt.ylabel('Número de Suicídios', fontsize=13)
plt.title('Série Temporal de Suicídios no Mundo (1965-2020)', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/time_series_suicidios_global.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo em: images/time_series_suicidios_global.png")

# Estatísticas globais
print("\n=== Estatísticas Globais ===")
print(f"Período analisado: {df_global['Year'].min():.0f} - {df_global['Year'].max():.0f}")
print(f"Total de suicídios registrados: {df_global['No of Suicides'].sum():,.0f}")
print(f"Média anual global: {df_global['No of Suicides'].mean():,.0f}")
print(f"Ano com mais suicídios: {df_global.loc[df_global['No of Suicides'].idxmax(), 'Year']:.0f} ({df_global['No of Suicides'].max():,.0f})")
print(f"Ano com menos suicídios: {df_global.loc[df_global['No of Suicides'].idxmin(), 'Year']:.0f} ({df_global['No of Suicides'].min():,.0f})")

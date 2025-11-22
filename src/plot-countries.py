import pandas as pd
import matplotlib.pyplot as plt
import os

# Criar diretório de imagens se não existir
os.makedirs('images', exist_ok=True)

# Carregar dados
df = pd.read_csv('data/combined_processed_data.csv')

# Filtrar dados dos países
df_usa = df[df['Country Name'] == 'United States of America'].groupby('Year')['No of Suicides'].sum().reset_index()
df_japan = df[df['Country Name'] == 'Japan'].groupby('Year')['No of Suicides'].sum().reset_index()

# Estados Unidos
plt.figure(figsize=(14, 7))
plt.plot(df_usa['Year'], df_usa['No of Suicides'], marker='o', linewidth=2, markersize=5, color='#3498db')
plt.xlabel('Ano', fontsize=13)
plt.ylabel('Número de Suicídios', fontsize=13)
plt.title('Série Temporal de Suicídios nos Estados Unidos', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/time_series_usa.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo em: images/time_series_usa.png")

# Japão
plt.figure(figsize=(14, 7))
plt.plot(df_japan['Year'], df_japan['No of Suicides'], marker='o', linewidth=2, markersize=5, color='#e74c3c')
plt.xlabel('Ano', fontsize=13)
plt.ylabel('Número de Suicídios', fontsize=13)
plt.title('Série Temporal de Suicídios no Japão', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/time_series_japan.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo em: images/time_series_japan.png")

# Estatísticas
print("\n=== Estatísticas por País ===")
for nome, dados in [("Estados Unidos", df_usa), ("Japão", df_japan)]:
    if len(dados) > 0:
        print(f"\n{nome}:")
        print(f"  Período: {dados['Year'].min():.0f} - {dados['Year'].max():.0f}")
        print(f"  Total: {dados['No of Suicides'].sum():,.0f}")
        print(f"  Média anual: {dados['No of Suicides'].mean():,.0f}")
        print(f"  Máximo: {dados['No of Suicides'].max():,.0f} ({dados.loc[dados['No of Suicides'].idxmax(), 'Year']:.0f})")

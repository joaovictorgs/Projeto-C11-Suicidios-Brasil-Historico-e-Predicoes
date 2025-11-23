import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from data_processing import process_japan_data


def analyze_japan_decomposition():
    # Criar diretório se não existir
    os.makedirs('images/decomposition-japan', exist_ok=True)
    
    df_japan = process_japan_data('data/combined_processed_data.csv')
    
    df_japan['Year'] = pd.to_datetime(df_japan['Year'], format='%Y')
    df_japan.set_index('Year', inplace=True)
    
    series = df_japan['No of Suicides']
    
    decomposition = seasonal_decompose(series, model='additive', period=12)
    
    # Gráfico 1: Observado
    plt.figure(figsize=(14, 5))
    decomposition.observed.plot(color='#2c3e50', linewidth=2)
    plt.ylabel('Observado', fontsize=12)
    plt.title('Série Observada - Suicídios no Japão', fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/decomposition-japan/observed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Tendência
    plt.figure(figsize=(14, 5))
    decomposition.trend.plot(color='#e74c3c', linewidth=2)
    plt.ylabel('Tendência', fontsize=12)
    plt.title('Tendência - Suicídios no Japão', fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/decomposition-japan/trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: Sazonalidade
    plt.figure(figsize=(14, 5))
    decomposition.seasonal.plot(color='#3498db', linewidth=2)
    plt.ylabel('Sazonalidade', fontsize=12)
    plt.title('Sazonalidade - Suicídios no Japão', fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/decomposition-japan/seasonal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 4: Resíduo
    plt.figure(figsize=(14, 5))
    decomposition.resid.plot(color='#95a5a6', linewidth=1)
    plt.ylabel('Resíduo', fontsize=12)
    plt.title('Resíduo - Suicídios no Japão', fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/decomposition-japan/residual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Análise de Decomposição - Japão")
    print("=" * 50)
    print("\nA série possui Tendência? Sim, tendência não linear")
    print("Tipo: Crescente de 1950 até ~1998 (pico ~31.000), depois decrescente até 2020")
    print("\nA série possui Sazonalidade? Sim")
    print("Período: Aproximadamente 12-15 anos (ciclos regulares visíveis)")
    print("\nA série apresenta um Ciclo? Sim")
    print("Razão: Possíveis fatores socioeconômicos, crises econômicas,")
    print("       mudanças culturais e políticas de saúde mental no Japão")
    print("\nGráficos salvos em: images/decomposition-japan/")
    print("  - observed.png")
    print("  - trend.png")
    print("  - seasonal.png")
    print("  - residual.png")


if __name__ == "__main__":
    analyze_japan_decomposition()

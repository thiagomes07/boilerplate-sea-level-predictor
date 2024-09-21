import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Leitura dos dados do arquivo CSV
    df = pd.read_csv('epa-sea-level.csv')

    # Criação do gráfico de dispersão (scatter plot)
    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'])

    # Criação da primeira linha de regressão linear (usando todos os dados)
    slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    
    # Geração de anos para a previsão até 2050
    years_extended = pd.Series(range(1880, 2051))
    sea_levels_predicted = intercept + slope * years_extended

    # Plotagem da primeira linha de melhor ajuste
    plt.plot(years_extended, sea_levels_predicted, color='red', label='Fit line 1880-2050')

    # Criação da segunda linha de regressão linear (usando dados de 2000 em diante)
    df_2000 = df[df['Year'] >= 2000]
    slope_2000, intercept_2000, r_value_2000, p_value_2000, std_err_2000 = linregress(df_2000['Year'], df_2000['CSIRO Adjusted Sea Level'])
    
    # Geração de anos para a previsão da segunda linha até 2050
    years_extended_2000 = pd.Series(range(2000, 2051))
    sea_levels_predicted_2000 = intercept_2000 + slope_2000 * years_extended_2000

    # Plotagem da segunda linha de melhor ajuste
    plt.plot(years_extended_2000, sea_levels_predicted_2000, color='green', label='Fit line 2000-2050')

    # Adição de rótulos e título
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')

    # Exibir a legenda
    plt.legend()

    # Salvar o gráfico
    plt.savefig('sea_level_plot.png')
    
    # Retornar o eixo do gráfico para teste
    return plt.gca()
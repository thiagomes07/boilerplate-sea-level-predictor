import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import numpy as np

# Predição usando regressão simples
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


# Predição usando o Scikit-Learn
def draw_plot_sklearn():
    # Ler os dados do arquivo CSV
    data = pd.read_csv('epa-sea-level.csv')

    # Criar o scatter plot dos dados
    plt.scatter(data['Year'], data['CSIRO Adjusted Sea Level'])

    # Modelo 1: Usar todos os dados para a primeira linha de melhor ajuste
    X_all = data['Year'].values.reshape(-1, 1)
    y_all = data['CSIRO Adjusted Sea Level'].values

    model_all = LinearRegression()
    model_all.fit(X_all, y_all)

    years_extended = np.arange(1880, 2051).reshape(-1, 1)
    sea_level_pred_all = model_all.predict(years_extended)

    plt.plot(years_extended, sea_level_pred_all, label="Best Fit (All Data)", color='r')

    # Modelo 2: Usar dados a partir de 2000
    recent_data = data[data['Year'] >= 2000]
    X_recent = recent_data['Year'].values.reshape(-1, 1)
    y_recent = recent_data['CSIRO Adjusted Sea Level'].values

    model_recent = LinearRegression()
    model_recent.fit(X_recent, y_recent)

    sea_level_pred_recent = model_recent.predict(years_extended)
    plt.plot(years_extended, sea_level_pred_recent, label="Best Fit (2000 onwards)", color='g')

    # Adicionar rótulos e título
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    plt.legend()

    # Salvar o gráfico
    plt.savefig('sea_level_plot_sklearn.png')
    return plt.gca()

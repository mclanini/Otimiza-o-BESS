import pandas as pd
from gera_grafico_FPO import gera_graficos_alocacao_bateria

DF = pd.read_csv(
    'resultados_FPO_alocacao_com_perdas.csv',
    sep=';',
    decimal=',',
    encoding='utf-8-sig'
)

gera_graficos_alocacao_bateria(
    DF, filename='comparativo_alocacao_bateria_com_perdas.html'
)
print('Arquivos gerados: resultados_FPO_alocacao_com_perdas.csv e '
      'comparativo_alocacao_bateria_com_perdas.html')
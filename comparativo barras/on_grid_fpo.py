"""Comparação da localização da bateria com perdas sempre habilitadas.

Execute: python on_grid_fpo_teste.py
Requer: FP_files/NE39.PWF, scripts.constants e pyscipopt (projeto original).
"""
from fpo_functions import read_pwf, get_Bl_matrix, build_fpo_model_pyscipopt_V2
from gera_grafico_FPO import gera_graficos_alocacao_bateria

import numpy as np
import pandas as pd

# 1. Dados e modificações originais da carga
file_data = read_pwf('./FP_files/NE39.PWF')
DBAR_ref = file_data['DBAR']
DBAR = DBAR_ref.copy()
DLIN = file_data['DLIN']
DGER = file_data['DGER']

for numero_barra in (4, 8, 20, 39):
    DBAR.loc[DBAR['No'] == numero_barra, 'Pl'] *= 2

NBAR = len(DBAR)
SLACK = np.where(DBAR['T'].to_numpy() == 2)[0]
Bl = get_Bl_matrix(NBAR, DLIN, SLACK)

# 2. Configuração do estudo
LIST_BAT_BUS = [1, 9, 19, 29]
LIST_PMAX_BAT = np.arange(1e-12, 75.5, 0.5).tolist()
SBASE = 100.0
FBASE = 60.0

barras_validas = set(DBAR['No'].astype(int))
faltantes = set(LIST_BAT_BUS) - barras_validas
if faltantes:
    raise ValueError(f'Barras candidatas não encontradas no DBAR: {sorted(faltantes)}')

resultados = []

# 3. Mesmo sistema e mesma capacidade para cada barra; sempre com perdas.
for bat_bus in LIST_BAT_BUS:
    for Pmax_bat in LIST_PMAX_BAT:
        R_bat = 1.0 / (20 * Pmax_bat)  # mesma regra de droop do original
        model, theta, Pg, load_shed, f, P_bat = build_fpo_model_pyscipopt_V2(
            DLIN, DBAR, DGER, Bl, SLACK,
            bat_bus=bat_bus,
            Pmax_bat=Pmax_bat,
            R_bat=R_bat,
            DBAR_ref=DBAR_ref,
            sbase=SBASE,
            fbase=FBASE,
            considerar_perdas=True,  # SEMPRE com perdas no balanço
        )
        model.optimize()
        status = str(model.getStatus())

        linha = {
            'Barra Bateria': bat_bus,
            'Capacidade Bateria (p.u.)': Pmax_bat,
            'Capacidade Bateria (MW)': Pmax_bat * SBASE,
            'Cenário': 'Com perdas',
            'Status': status,
            'Corte de Carga (MW)': np.nan,
            'Função Objetivo': np.nan,
            'Frequência (Hz)': np.nan,
            'Potência Bateria (MW)': np.nan,
            'Perdas Totais (MW)': np.nan,
        }
        if model.getNSols() > 0:
            valores_theta = {i: model.getVal(theta[i]) for i in range(NBAR)}
            perdas_pu = 0.0
            # Mesma fórmula utilizada nas restrições em fpo_functions_teste.py:
            # g = r/(r^2+x^2), perda = g*(theta_de-theta_para)^2.
            indice = {int(bus): i for i, bus in enumerate(DBAR['No'])}
            for _, ramo in DLIN.iterrows():
                r = float(ramo['R%']) / 100.0
                x = float(ramo['X%']) / 100.0
                if r == 0 and x == 0:
                    raise ValueError('Ramo com R=X=0; perda indefinida.')
                g = r / (r * r + x * x)
                i = indice[int(ramo['De'])]
                j = indice[int(ramo['Pa'])]
                perdas_pu += g * (valores_theta[i] - valores_theta[j]) ** 2

            linha.update({
                'Corte de Carga (MW)': SBASE * sum(model.getVal(load_shed[i]) for i in range(NBAR)),
                'Função Objetivo': model.getObjVal(),
                'Frequência (Hz)': FBASE * model.getVal(f),
                'Potência Bateria (MW)': SBASE * model.getVal(P_bat),
                'Perdas Totais (MW)': SBASE * perdas_pu,
            })
            print(f'Barra={bat_bus:>2} | C={Pmax_bat:8.4f} p.u. | '
                  f'{status} | Perdas={linha["Perdas Totais (MW)"]:.4f} MW | '
                  f'Corte={linha["Corte de Carga (MW)"]:.4f} MW')
        else:
            print(f'Barra={bat_bus:>2} | C={Pmax_bat:8.4f} p.u. | '
                  f'{status} | Sem solução')
        resultados.append(linha)

# 4. Exportações e gráficos
DF = pd.DataFrame(resultados)
DF.to_csv('resultados_FPO_alocacao_com_perdas.csv', sep=';', decimal=',',
          index=False, encoding='utf-8-sig')

gera_graficos_alocacao_bateria(
    DF, filename='comparativo_alocacao_bateria_com_perdas.html'
)
print('Arquivos gerados: resultados_FPO_alocacao_com_perdas.csv e '
      'comparativo_alocacao_bateria_com_perdas.html')

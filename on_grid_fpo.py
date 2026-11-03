from fpo_functions_teste import read_pwf, get_Bl_matrix, build_fpo_model_pyscipopt_V2
from gera_grafico_FPO_teste import gera_gráficos_curva_new

import numpy as np
import pandas as pd

file_data = read_pwf('./FP_files/NE39.PWF')
DBAR_ref = file_data['DBAR']
DBAR = DBAR_ref.copy()
DLIN = file_data['DLIN']
DGER = file_data['DGER']

# Mantém o mesmo aumento de carga do caso original.
for numero_barra in (4, 8, 20, 39):
    DBAR.loc[DBAR['No'] == numero_barra, 'Pl'] *= 2

NBAR = len(DBAR)
SLACK = np.where(DBAR['T'].to_numpy() == 2)[0]
Bl = get_Bl_matrix(NBAR, DLIN, SLACK)

bat_bus = 9
list_Pmax_bat = np.arange(1e-12, 75.5, 0.5).tolist()
sbase = 100.0
resultados = []

for Pmax_bat in list_Pmax_bat:
    R_bat = 1 / (20 * Pmax_bat)

    for considerar_perdas in (False, True):
        cenario = 'Com perdas' if considerar_perdas else 'Sem perdas'
        model, theta, Pg, load_shed, f, P_bat = build_fpo_model_pyscipopt_V2(
            DLIN, DBAR, DGER, Bl, SLACK,
            bat_bus=bat_bus,
            Pmax_bat=Pmax_bat,
            R_bat=R_bat,
            DBAR_ref=DBAR_ref,
            sbase=sbase,
            considerar_perdas=considerar_perdas,
        )
        model.optimize()
        status = str(model.getStatus())

        linha = {
            'Capacidade Bateria (p.u.)': Pmax_bat,
            'Cenário': cenario,
            'Status': status,
            'Corte de Carga': np.nan,
            'Função Objetivo': np.nan,
        }
        if model.getNSols() > 0:
            # Corte de carga TOTAL, em MW (não o maior corte individual).
            linha['Corte de Carga'] = sbase * sum(
                model.getVal(load_shed[i]) for i in range(NBAR)
            )
            linha['Função Objetivo'] = model.getObjVal()
            print(f'{cenario} | Capacidade={Pmax_bat:.4f} | '
                  f'Status={status} | Corte={linha["Corte de Carga"]:.4f} MW | '
                  f'FOB={linha["Função Objetivo"]:.4f}')
        else:
            print(f'{cenario} | Capacidade={Pmax_bat:.4f} | '
                  f'Status={status} | Nenhuma solução disponível')
        resultados.append(linha)

# Cada capacidade produz duas linhas, uma para cada cenário.
df = pd.DataFrame(resultados)
df.to_csv('resultados_FPO_com_sem_perdas.csv', sep=';', decimal=',', index=False,
          encoding='utf-8-sig')
gera_gráficos_curva_new(df)

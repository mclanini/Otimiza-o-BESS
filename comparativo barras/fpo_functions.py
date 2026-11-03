# from on_grid_fpo import DGER
import pandas as pd
import numpy as np

from scripts.constants import PWF_DATA_DICT
from pyscipopt import Model, quicksum


def read_pwf(file_path: str) -> dict[str, pd.DataFrame]:
    """Função que lê o arquivo PWF informado e retorna em dataframes
    todos os blocos de dados do PWF (DBAR, DLIN, DGER, etc)

    Args:
        file_path (str): Caminho para o arquivo .pwf

    Returns:
        dict[str, pd.DataFrame]: Dicionário onde cada chave é um bloco
        de dados do arquivo e o valor é um dataframe com os valores
    """

    result = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for data_title in PWF_DATA_DICT.keys():
        block_datas = PWF_DATA_DICT[data_title].keys()

        start_data_idx = next((i for i, line in enumerate(lines) if line.strip() == data_title), None)
        if start_data_idx is None:
            continue

        end_data_idx = (start_data_idx + 2) + lines[start_data_idx + 2:].index("99999\n")

        block_df = None
        for i in range(start_data_idx + 2, end_data_idx):
            line = lines[i]
            line_dict = {}

            for data in block_datas:
                begin_data_idx, end_data_idx = PWF_DATA_DICT[data_title][data]["index"]
                column_data = line[begin_data_idx: end_data_idx].strip()

                if "\n" in column_data:
                    break
                if column_data == '':
                    column_data = PWF_DATA_DICT[data_title][data]["default_value"]

                data_type = PWF_DATA_DICT[data_title][data]["type"]
                line_dict[data] = data_type(column_data)

            block_df = pd.concat([block_df, pd.DataFrame([line_dict])], ignore_index=True)

        result[data_title] = block_df

    
    return result


def get_Bl_matrix(n_bar: int, DLIN: pd.DataFrame, SLACK: np.ndarray) -> np.ndarray:
    """Calcula a matriz B' dados NBAR e DLIN

    Args:
        DBAR (pd.DataFrame): Dataframe de DBAR
        dlin (pd.DataFrame): Dataframe de DLIN

    Returns:
        np.ndarray: Matriz B' do numpy
    """

    # Inicializa B'
    Bl = np.zeros((n_bar, n_bar))

    # Obter cada variavel de DLIN
    from_bus = [int(value) for value in list(DLIN["De"])]
    to_bus = [int(value) for value in list(DLIN["Pa"])]
    x = [float(value)/100 for value in list(DLIN["X%"])]

    for i in range(len(from_bus)):
        # Diagonal
        Bl[from_bus[i] - 1][from_bus[i] - 1] += 1/x[i]
        Bl[to_bus[i] - 1][to_bus[i] - 1] += 1/x[i]

        # Fora da diagonal
        Bl[from_bus[i] - 1][to_bus[i] - 1] -= 1/x[i]
        Bl[to_bus[i] - 1][from_bus[i] - 1] -= 1/x[i]

    Bl[SLACK, SLACK] = 1e9

    return Bl

def build_fpo_model_pyscipopt_V2(DLIN, DBAR, DGER, Bl, SLACK, bat_bus=None, Pmax_bat=0, R_bat=1, DBAR_ref=None, sbase=100.0, fbase=60.0, f_min_hz=59.5, f_max_hz=61.0, max_load_shed_frac=1.0, considerar_perdas=True):
    # ============================================================
    # Inicialização
    # ============================================================
    model = Model("FPO_Frequency_Control_BESS")
    model.setIntParam("display/verblevel", 0)

    DBAR = DBAR.reset_index(drop=True).copy()
    DGER = DGER.reset_index(drop=True).copy()
    DLIN = DLIN.reset_index(drop=True).copy()

    if DBAR_ref is None:
        DBAR_ref = DBAR.copy()
    else:
        DBAR_ref = DBAR_ref.reset_index(drop=True).copy()

    DBAR["No"] = DBAR["No"].astype(int)
    DBAR_ref["No"] = DBAR_ref["No"].astype(int)
    DGER["No"] = DGER["No"].astype(int)

    DLIN["De"] = DLIN["De"].astype(int)
    DLIN["Pa"] = DLIN["Pa"].astype(int)
    DLIN["R%"] = DLIN["R%"].astype(float)

    Nbar = DBAR.shape[0]

    Nlin = DLIN.shape[0]

    perda_barra = {i: 0 for i in range(Nbar)}

    bus_to_idx = {int(bus): idx for idx, bus in enumerate(DBAR["No"].values)}

    ref_by_bus = DBAR_ref.set_index("No")
    DGER_indexed = DGER.set_index("No")

    slack_idx = int(SLACK[0])

    # ============================================================
    # Validação da bateria
    # ============================================================
    if bat_bus is not None:
        bat_bus = int(bat_bus)

        if bat_bus not in bus_to_idx:
            raise ValueError(f"bat_bus = {bat_bus} não existe no DBAR.")

        if Pmax_bat < 0:
            raise ValueError(f"Pmax_bat inválido: {Pmax_bat}")

        if R_bat <= 0:
            raise ValueError(f"R_bat inválido: {R_bat}")

    # ============================================================
    # Cargas atuais
    # ============================================================
    Pd = DBAR["Pl"].to_numpy(dtype=float) / sbase

    # ============================================================
    # Geração de referência do caso base
    # ============================================================
    Pg0_ref = np.zeros(Nbar, dtype=float)
    Pd_ref = np.zeros(Nbar, dtype=float)

    for i in range(Nbar):
        bus = int(DBAR.loc[i, "No"])

        if bus not in ref_by_bus.index:
            raise ValueError(f"Barra {bus} não encontrada no DBAR_ref.")

        Pg0_ref[i] = float(ref_by_bus.loc[bus, "Pg"]) / sbase
        Pd_ref[i] = float(ref_by_bus.loc[bus, "Pl"]) / sbase

    # Ajuste da geração da slack no caso base DC sem perdas.
    mismatch_ref = np.sum(Pd_ref) - np.sum(Pg0_ref)
    Pg0_ref[slack_idx] += mismatch_ref

    # ============================================================
    # Identificação dos geradores com governor
    # ============================================================
    gen_buses = list(map(int, DGER["No"].values))
    gen_buses_set = set(gen_buses)

    # ============================================================
    # Variáveis
    # ============================================================

    # Ângulos
    theta = {i: model.addVar(lb=-1e3, ub=1e3, name=f"theta_{i}") for i in range(Nbar)}

    # Frequência em pu
    f = model.addVar(lb=f_min_hz / fbase, ub=f_max_hz / fbase, name="freq")

    # Corte de carga
    load_cut = {i: model.addVar(lb=0.0, ub=max_load_shed_frac * Pd[i], name=f"ls_{i}") for i in range(Nbar)}

    # Geração
    Pg = {}

    for i in range(Nbar):
        bus = int(DBAR.loc[i, "No"])

        if bus in gen_buses_set:
            Pmin = 0.0
            Pmax = 1e4

            if "Pmn" in DGER_indexed.columns:
                Pmin_raw = float(DGER_indexed.loc[bus, "Pmn"])
                if Pmin_raw > 0:
                    Pmin = Pmin_raw / sbase

            if "Pmx" in DGER_indexed.columns:
                Pmax_raw = float(DGER_indexed.loc[bus, "Pmx"])
                if 0 < Pmax_raw < 9999:
                    Pmax = Pmax_raw / sbase

            Pg[i] = model.addVar(lb=Pmin, ub=Pmax, name=f"Pg_{i}")
        else:
            Pg[i] = Pg0_ref[i]

    # ============================================================
    # Variável da bateria
    # ============================================================
    if bat_bus is not None:
        P_bat = model.addVar(lb=-Pmax_bat, ub=Pmax_bat, name="P_bat")
    else:
        P_bat = None

    # As perdas só entram no balanço do cenário com perdas.
    # No cenário sem perdas, nenhuma expressão de perda é construída,
    # e DLIN (inclusive R%) permanece inalterado.
    if considerar_perdas:
        for k in range(Nlin):
            de = bus_to_idx[int(DLIN.loc[k, "De"])]
            para = bus_to_idx[int(DLIN.loc[k, "Pa"])]
            r = float(DLIN.loc[k, "R%"]) / 100
            x = float(DLIN.loc[k, "X%"]) / 100
            g = r / (r**2 + x**2)
            # r = 1/(float(DLIN.loc[k, "R%"]) / 100)
            theta_km = theta[de] - theta[para]
            perda = g * theta_km**2
            perda_barra[de] += 0.5 * perda
            perda_barra[para] += 0.5 * perda


    # ============================================================
    # Balanço nodal
    # ============================================================
    for i in range(Nbar):
        bus = int(DBAR.loc[i, "No"])

        bat_term = 0.0

        if bat_bus is not None and bus == bat_bus:
            bat_term = P_bat

        balanco = Pg[i] + bat_term - Pd[i] + load_cut[i]
        if considerar_perdas:
            balanco -= perda_barra[i]

        model.addCons(
            quicksum(Bl[i, j] * theta[j] for j in range(Nbar)) == balanco,
            name=f"power_balance_{i}"
        )

    # ============================================================
    # Governor dos geradores
    # ============================================================
    for bus in gen_buses:
        if bus not in bus_to_idx:
            raise ValueError(f"Gerador informado em DGER na barra {bus}, mas essa barra não existe em DBAR.")

        idx = bus_to_idx[bus]

        R = float(DGER_indexed.loc[bus, "R"])/100

        if R <= 0:
            raise ValueError(f"R inválido na barra {bus}: R = {R}")

        # R em pu na base do sistema:
        # Pg_i = Pg0_i + (1 - f) / R_i
        model.addCons(Pg[idx] == Pg0_ref[idx] + (1 - f) / R, name=f"droop_{bus}")

    # ============================================================
    # Referência angular
    # ============================================================
    for s in SLACK:
        model.addCons(theta[int(s)] == 0.0, name=f"theta_slack_{int(s)}")

    # ============================================================
    # Droop da bateria
    # ============================================================
    custo_bat = model.addVar(lb=0.0, name="custo_bat")
    if P_bat is not None:
        model.addCons(P_bat == (1 - f) / R_bat, name="battery_droop")
        model.addCons(custo_bat >= P_bat**2,name="custo_quadratico_bateria")
    else:
        model.addCons(custo_bat == 0.0)

    # ============================================================
    # Função objetivo: custo unitário de corte de carga
    # ============================================================
    model.setObjective((sbase * quicksum(load_cut[i] for i in range(Nbar)))*1000 + (custo_bat*10 + Pmax_bat*100), "minimize")

    return model, theta, Pg, load_cut, f, P_bat
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import webbrowser
import os
import re
import time

from scripts.constants import PWF_DATA_DICT
from math import cos, sin, radians


# -------------------- FUNCOES MODELO LINEAR -------------------- #
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


def get_linear_model(DBAR: pd.DataFrame, DGER: pd.DataFrame, Bl: np.ndarray, SLACK: np.ndarray, pr_reg_busses: list) -> tuple[np.ndarray, np.ndarray]:
    """Função que gera as matrizes base do governor

    Args:
        DBAR (pd.DataFrame): Dataframe de DBAR
        DGER (pd.DataFrame): Dataframe de DGER
        Bl (np.ndarray): Matriz B linha
        SLACK (np.ndarray): Vetor com indices das barras slack

    Returns:
        tuple[np.ndarray, np.ndarray]: tupla com a matriz Y e A para a resolucao X = Y/A
    """

    # Dados do sistema
    Bger = DGER["No"].to_numpy() - 1
    Nbar = DBAR.shape[0]

    # Criação do vetor Y
    Pl_neg = (np.where(DBAR["No"].isin(pr_reg_busses), -DBAR["Pl"], DBAR["Pg"] - DBAR["Pl"]))/100
    # Pl_neg = -(DBAR["Pl"].to_numpy() / 100)
    # Pl_neg[SLACK] = 0

    # Filtrar apenas barras com regulação primária
    df_reg = DBAR[DBAR["No"].isin(pr_reg_busses)]

    # Vetor de Psch + f/R
    P_aux = (df_reg["Pg"]/100 + 1 / (df_reg["No"].map(DGER.set_index("No")["R"])/100)).to_numpy()
    # P_aux = DBAR.loc[Bger, "Pg"].to_numpy() / 100
    # P_aux = P_aux + 1 / (DGER["R"].to_numpy() / 100)

    # Vetor de θ refêrencia
    Theta_aux = np.zeros(len(SLACK))

    # Vetor Y resultante (-Pd; Pf; θref; f)
    Y = np.concatenate([Pl_neg, P_aux, Theta_aux])

    # Criação da matriz A do sistema Y = AX (expansão do P = B'θ via governor). A é formada por [Bl E F; G H I; J K L]
    # Matriz Bl com Big Number
    Bl = Bl.copy()
    Bl[SLACK, SLACK] = 1e9

    # Matriz E
    E = np.zeros((Nbar, len(pr_reg_busses)))
    E[np.array(pr_reg_busses) - 1, np.arange(len(pr_reg_busses))] = -1

    # Matriz E - Identidade negativa
    # E = -np.eye(Nbar)
    # mask = np.ones(Nbar, dtype=bool)
    # mask[Bger] = False
    # E = E[:, ~mask]

    # Matriz F - nula
    F = np.zeros((E.shape[0], 1))

    # Matriz G - nula
    G = np.zeros((P_aux.shape[0], Bl.shape[1]))

    # Matriz H - identidade
    H = np.eye(E.shape[1])

    # Matriz I - 1/Ri
    I = (1 / (df_reg["No"].map(DGER.set_index("No")["R"])/100)).to_numpy().reshape(-1, 1)
    # I = 1.0 / (DGER["R"].to_numpy() / 100)
    # I = I.reshape(-1, 1)

    # Matriz J - nula porem com 1 no(s) θ refêrencia(s)
    J = np.zeros((1, Bl.shape[1]))
    J[0, SLACK] = 1

    # Matriz K - nula
    K = np.zeros((1, E.shape[1]))

    # Matriz L - nula
    L = np.zeros((1, 1))

    # Matriz A total
    A = np.block([
        [Bl, E, F],
        [G, H, I],
        [J, K, L]
    ])

    return (Y, A)


def solve_linear_loss(A: np.ndarray, Y: np.ndarray, NBAR: int, DLIN: pd.DataFrame, max_iter: int=100, tol: float=1e-6) -> np.ndarray:
    """Resolve o flow linear considerando perdas no processo iterativo

    Args:
        A (np.ndarray): Matriz A
        Y (np.ndarray): Matriz de termos independentes Y
        NBAR (int): Número de barras
        DLIN (pd.DataFrame): Matriz DLIN
        max_iter (int, optional): Número máximo de iterações. Defaults to 20.
        tol (float, optional): Tolerancia entre a variação da perda total para critério de parada. Defaults to 1e-6.
    """

    if not any(DLIN["R%"]):
        return np.linalg.solve(A, Y)

    Y_copy = Y.copy().astype(float)
    total_loss = 0

    for _ in range(max_iter):
        result = np.linalg.solve(A, Y_copy)
        Thetas = result[0:NBAR]

        line_loss = []
        bar_loss = np.zeros_like(Y_copy)

        for _, row in DLIN.iterrows():
            k = row["De"] - 1
            m = row["Pa"] - 1
            R = row["R%"] / 100
            X = row["X%"] / 100

            Gkm = R/(R**2 + X**2) if R != 0 else 0

            loss_km = Gkm * (Thetas[k] - Thetas[m])**2
            line_loss.append(loss_km)

            bar_loss[k] -= loss_km / 2
            bar_loss[m] -= loss_km / 2

        Y_copy = Y + bar_loss

        curr_total_loss = np.sum(line_loss)

        if abs(curr_total_loss - total_loss) < tol:
            return result
        
        total_loss = curr_total_loss

    raise RuntimeError("Iteração para estimativas de perdas não convergiu")


def solve_governor_power_flow_linear(Bl: np.ndarray, DBAR: pd.DataFrame, DGER: pd.DataFrame, DLIN: pd.DataFrame, SLACK: np.ndarray, Pbss_max: float, Pbss_min: float, pr_reg_busses: list, loss_estimation: bool=True) -> np.ndarray:
    """Função que resolve o FP linear com as baterias

    Args:
        Bl (np.ndarray): Matriz B'
        DBAR (pd.DataFrame): Dataframe dos dados de barra
        DGER (pd.DataFrame): Dataframe dos dados de gerador
        DLIN (pd.DataFrame): Dataframe dos dados de linha
        SLACK (np.ndarray): Indices das barras slack
        Pbss_max (float): Valor maximo que Pbss pode assumir
        Pbss_min (float): Valor minimo que Pbss pode assumir
        mode (string): Modo de controle "ON" para on-grid, "OFF" para off-grid
        tol (float): Tolerância em mHz de variação de frequencia para o modo on-grid

    Returns:
        p.ndarray: Vetor dos resultados e capacidade da bateria final
    """

    NBAR = len(DBAR)

    Y, A = get_linear_model(DBAR, DGER, Bl, SLACK, pr_reg_busses) 

    loss = any(DLIN["R%"])

    f_esp = 1

    # Acrescentando as matrizes da frequencia
    M = np.zeros((A.shape[0], 1))
    M[2, 0] = -1 # Bateria na barra 3

    N = np.zeros((1, A.shape[1] + 1))
    N[0, -2] = 1
    N[0, -1] = 0

    A = np.hstack([A, M])
    A = np.vstack([A, N])

    # Adiciona a frequencia especificada
    Y = np.concatenate([Y, [f_esp]])

    start = time.time()
    # Resolução do sistema Y = AX
    if loss_estimation:
        X = solve_linear_loss(A, Y, NBAR, DLIN)
    else:
        X = np.linalg.solve(A, Y)

    Pbss = X[-1]

    if Pbss < Pbss_min or Pbss > Pbss_max:
        Y[-1] = np.clip(Pbss, Pbss_min, Pbss_max)

        A[-1, :] = 0
        A[-1, -1] = 1
        X = np.linalg.solve(A, Y)

    end = time.time()

    # return np.concatenate((np.ones(NBAR), X, [pbss_level]))
    return np.concatenate((np.ones(NBAR), X)), end - start


def calculate_reactive_power(DLIN: pd.DataFrame, DBAR: pd.DataFrame, DGER: pd.DataFrame, flow_solution: np.ndarray) -> np.ndarray:
    """Calcula a potencia reativa dos geradores presentes a partir
    dos valores encontrados na solução do problema matricial

    Args:
        DLIN (pd.DataFrame): Dataframe dos dados de linha
        DBAR (pd.DataFrame): Dataframe dos dados de barra
        DGER (pd.DataFrame): Dataframe dos dados de geração
        flow_solution (np.ndarray): Vetor com as soluções do problema original

    Returns:
        np.ndarray: Vetor com os Qgs de cada gerador
    """
    NBAR = len(DBAR)

    Y_bus = get_ybus_matrix(DLIN, DBAR)

    theta = flow_solution[NBAR:2 * NBAR]

    G = Y_bus.real
    B = Y_bus.imag

    Q_inj = np.zeros(NBAR)

    for k in range(NBAR):
        for m in range(NBAR):
            theta_km = theta[k] - theta[m]
            Q_inj[k] += G[k, m] * np.sin(theta_km) - B[k, m] * np.cos(theta_km)

    Qg_calc = []
    for _, ger in DGER.iterrows():
        bus = int(ger["No"]) - 1
        Ql = DBAR.loc[DBAR["No"] == bus + 1, "Ql"].values[0] / 100
        Qg_calc.append(Q_inj[bus] + Ql)

    return Qg_calc


def get_linear_result(path_file: str, pbss_capacity: float, load_multiplier: float, loss_estimation: bool=True) -> pd.DataFrame:
    """
    Função que calcula o flow linear e retorna o datframe com resultados

    Args:
        path_file (str): Diretorio do caminho do arquivo pwf
        pbss_capacity (float): Capacidade do banco em PU
        load_multiplier (float): multiplicador da carga

    Returns:
        pd.DataFrame: Dataframe com resultados
    """

    file_data = read_pwf(path_file)
    DBAR = file_data.get("DBAR", None)
    DLIN = file_data.get("DLIN", None)
    DGER = file_data.get("DGER", None)
    DCCA = file_data.get("DCCA", None)
    DCGS = file_data.get("DCGS", None)
    DCGE = file_data.get("DCGE", None)

    # Alterando curva de carga
    DCCA["P"] *= load_multiplier
    DCCA["Q"] *= load_multiplier

    NBAR = len(DBAR)
    SLACK = np.where(DBAR["T"].to_numpy() == 2)[0]
    PR_REG_BUSSES = [3, 4]

    Bl = get_Bl_matrix(NBAR, DLIN, SLACK)

    # Inicializa DataFrame de resultados
    result = pd.DataFrame(columns=[f"VD {num}" for num in DBAR["No"]] + [f"TD {num}" for num in DBAR["No"]] + [f"PG Barra-{num}" for num in DGER["No"]] + ["Frequencia (Hz)", "PBSS", \
                        "Nivel Bateria (%)", "Fator de Carga", "Fator Geração Eólica", "Fator Geração Solar", "Ponto da Curva de Carga"] + [f"QG Barra-{num}" for num in DGER["No"]] + ["Iteracoes"])

    Pbss_level_percent = 100 # (%)
    Pbss_level = (Pbss_level_percent/100) * pbss_capacity
    Pbss_health_limit = 0.2 * pbss_capacity
    Pbss_max = pbss_capacity - Pbss_health_limit
    Pbss_min = - (pbss_capacity - Pbss_level)

    times = []
    
    for i in range(len(DCCA)):
        # Ajusta a potência com a curva de demanda
        DBAR_copy = DBAR.copy()

        pl_mult = DCCA["P"].iloc[i]
        ql_mult = DCCA["Q"].iloc[i]

        DBAR_copy["Pl"] = DBAR["Pl"] * pl_mult
        DBAR_copy["Ql"] = DBAR["Ql"] * ql_mult

        # Ajusta as gerações
        if DCGS is not None:
            BUS = int(DCGS.loc[i, "num"])
            pgs_mult = DCGS["P"].iloc[i]
            qgs_mult = DCGS["Q"].iloc[i]

            DBAR_copy.loc[DBAR.index[DBAR["No"] == BUS][0], "Pg"] *= pgs_mult
            # DBAR_copy.loc[DBAR.index[DBAR["No"] == BUS][0], "Qg"] *= q_mult

        if DCGE is not None:
            BUS = int(DCGE.loc[i, "num"])
            pge_mult = DCGE["P"].iloc[i]
            qge_mult = DCGE["Q"].iloc[i]

            DBAR_copy.loc[DBAR.index[DBAR["No"] == BUS][0], "Pg"] *= pge_mult
            # DBAR_copy.loc[DBAR.index[DBAR["No"] == BUS][0], "Qg"] *= q_mult

        # Resolve o fluxo de potencia linear
        flow_solution, time = solve_governor_power_flow_linear(Bl, DBAR_copy, DGER, DLIN, SLACK, Pbss_max, Pbss_min, PR_REG_BUSSES, loss_estimation)
        times.append(time)

        # Adicionado Pg das barras sem regulação primária
        gen_buses = np.sort(DGER["No"].to_numpy())
        Pg_base = (DBAR_copy.set_index("No").loc[gen_buses, "Pg"].to_numpy()) / 100
        Pg_pr = flow_solution[2*NBAR : 2*NBAR + len(PR_REG_BUSSES)]
        pos_pr = np.searchsorted(gen_buses, PR_REG_BUSSES)
        Pg_base[pos_pr] = Pg_pr
        flow_solution = np.concatenate([flow_solution[:2*NBAR], Pg_base, flow_solution[2*NBAR + len(PR_REG_BUSSES):]])


        # Atualizando status da bateria
        Pbss = flow_solution[-1]
        Pbss_level -= Pbss
        Pbss_max = Pbss_level - Pbss_health_limit
        Pbss_min = - (pbss_capacity - Pbss_level)

        # Calcular Qg
        Qg_calc = calculate_reactive_power(DLIN, DBAR_copy, DGER, flow_solution)

        sol = np.concatenate((flow_solution, [Pbss_level, pl_mult, pge_mult, pgs_mult, i], Qg_calc, [0]))

        result.loc[i] = sol

    print(f"Media de tempo linear {np.mean(times)}")

    # Convertendo valores
    result["Frequencia (Hz)"] *= 60

    for bar in range(NBAR):
        result[f"TD {bar + 1}"] *= (180/np.pi)

    result["Nivel Bateria (%)"] *= (100/pbss_capacity)

    return result


# ------------------ FUNCOES MODELO NAO LINEAR ------------------ #
def get_ybus_matrix(DLIN: pd.DataFrame, DBAR: pd.DataFrame) -> np.ndarray:
    n_bus = len(DBAR)
    n_lin = len(DLIN)

    # Inicialização da Ybarra
    Y_bus = np.zeros([n_bus, n_bus], dtype=complex)

    # Obter cada variável de DLIN
    from_busses = DLIN["De"]
    to_busses = DLIN["Pa"]
    R = DLIN["R%"]/100
    X = DLIN["X%"]/100
    B_shunt = DLIN["Mvar"]/200 # Divide por 2x100 pois é metade para cada barra
    tap = DLIN["Tap"]
    Sh = DBAR["Sh"]

    for i in range(n_lin):
        # Cálculo da diagonal
        Y_bus[from_busses[i] - 1, from_busses[i] - 1] += (1/complex(R[i], X[i]) + complex(0, B_shunt[i])) * (1/tap[i])**2
        Y_bus[to_busses[i] - 1, to_busses[i] - 1] += 1/complex(R[i], X[i]) + complex(0, B_shunt[i])

        # Calculo fora da diagonal
        Y_bus[from_busses[i] - 1, to_busses[i] - 1] -= (1/complex(R[i], X[i])) * (1/tap[i])
        Y_bus[to_busses[i] - 1, from_busses[i] - 1] -= (1/complex(R[i], X[i])) * (1/tap[i])

    # Soma o Sh na diagonal da barra
    for i in range(n_bus):
        Y_bus[i, i] += complex(0, Sh[i]/100)

    return Y_bus


def calc_ybarra(nbar, nlin, dlin, dbar):
    # Inicialização
    ybarra = np.zeros([nbar, nbar], dtype=complex)
    i = 0
    ykm = np.zeros([nlin, 1], dtype=complex)
    for lin in dlin:
        p = int(lin[0]) - 1
        q = int(lin[1]) - 1
        tap = float(lin[5])
        # Cálculos
        zkm = complex(lin[2], lin[3])
        bsh = complex(0, lin[4]/2.0)
        ykm[i] = 1/zkm
        if tap > 0.0:
            ybarra[p,p] += ykm[i] * 1/tap**2 + bsh #a**2
            ybarra[p,q] -= ykm[i] * 1/tap #-a
        else:
            ybarra[p,p] += ykm[i] + bsh
            ybarra[p,q] -= ykm[i]
        ybarra[q,q] += ykm[i] + bsh
        ybarra[q,p] = ybarra[p,q] #-a
        if tap > 0.0:
            ykm[i] = -ykm[i] * 1/tap
        else:
            ykm[i] = -ykm[i]
        i +=1
    
    # Adiciona contribuições dos capacitores
    for barra in dbar:
        num = int(barra[0]) - 1
        shunt = complex(0, barra[7])
        ybarra[num, num] += shunt
    return ybarra, ykm


def solver_newton(dbar, nbar, dlin, nlin, tol, max_iter, dger=None, f_esp=None, nbss=None, PbssMax=None, PbssMin=None, list_renovaveis=None):
    start = time.time()
    ybarra, ykm = calc_ybarra(nbar, nlin, dlin, dbar)

    # Matriz de Condutância Nodal
    G = ybarra.real
    # Matriz de Susceptância Nodal
    B = ybarra.imag

    # Verifica a barra de referência no sistema
    ref = [True if int(barra[1]) == 2 else False for barra in dbar]

    # Verifica se a barra é PV
    verifica_pv = [True if int(barra[1]) == 1 else False for barra in dbar]

    # Inicialização de variáveis para o cálculo
    # Representação de infinito
    big_num = 1.0e10
    # Potência final nas barras (Pg - Pl e Qg - Ql)
    pl = [barra[5] for barra in dbar]
    pf = [barra[3] - barra[5] for barra in dbar]
    qf = [barra[4] - barra[6] for barra in dbar]
    # Tensao
    vd = [barra[2] for barra in dbar]
    # Theta
    # td = [0.0] * nbar
    td = [radians(barra[8]) for barra in dbar]
    td_esp = td
    # de 
    nfrom = [int(lin[0]) for lin in dlin]
    # para
    nto = [int(lin[1]) for lin in dlin]
    # delta_p e delta_q
    delta_p = np.zeros([nbar, 1])
    delta_q = np.zeros([nbar, 1])

    # calculo da frequência
    if (dger is not None) and (f_esp is not None):
        # Filtrar onde a coluna OETGb (índice 1) é 1 ou 2
        nbarg = np.sum((dbar[:, 1] == 1) | (dbar[:, 1] == 2))
        nref = np.sum(dbar[:, 1] == 2)
        delta_yi = np.zeros([nbarg, 1])
        delta_y = np.zeros([nref, 1])
        filtro_geradoras = (dbar[:, 1] == 1) | (dbar[:, 1] == 2)
        dbar_g = dbar[filtro_geradoras]
        dbar_g = np.column_stack((dbar_g, dger))
        nogen = [int(barra[0]) for barra in dbar_g]
        pg_esp = [barra[3] for barra in dbar_g]
        f_calc = f_esp
        pgi = pg_esp
        r_est = [barra[10] for barra in dbar_g]

        n_bat = len(nbss)
        pbss = 0
        # PbssCap = 0.8
        # PbssMax = PbssCap
        warning_pbss = False

    # Guarda erro em máximo de cada iteração em uma lista
    erro_list = list()
    # Contador de iterações
    iter = 1
    # Flag convergência
    convergiu = False

    while iter <= max_iter:
        # Inicializa o vetor de potencia ativa e reativa calculadas
        p_calc = np.zeros([nbar, 1])
        q_calc = np.zeros([nbar, 1])

        # Prepara sistema de equacoes
        # elementos da diagonal
        for n in range(nbar):
            p_calc[n, 0] = + vd[n] ** 2 * G[n, n]
            q_calc[n, 0] = - vd[n] ** 2 * B[n, n]
        # elementos fora da diagonal
        i=0
        for linha in dlin:
            k = int(linha[0]) - 1 # "de"
            m = int(linha[1]) - 1 # "para"
            # Theta
            tkm = td[k] - td[m]
            tmk = -tkm
            # Condutância
            Gkm = ykm[i].real
            # Susceptância
            Bkm = ykm[i].imag
            
            p_calc[k, 0] += (vd[k] * vd[m]) * ((Gkm * cos(tkm)) + (Bkm * sin(tkm)))
            p_calc[m, 0] += (vd[m] * vd[k]) * ((Gkm * cos(tmk)) + (Bkm * sin(tmk)))

            q_calc[k, 0] += (vd[k] * vd[m]) * ((Gkm * sin(tkm)) - (Bkm * cos(tkm)))
            q_calc[m, 0] += (vd[m] * vd[k]) * ((Gkm * sin(tmk)) - (Bkm * cos(tmk)))
            i +=1
        
        # Atualiza pf
        if (dger is not None) and (f_esp is not None):
            for barra in dbar:
                n = int(barra[0]) - 1
                if (int(barra[1]) == 2) or (int(barra[1]) == 1):
                    x = nogen.index(n+1)
                    pf[n] = pgi[x] - pl[n]
                else:
                    pf[n] = pf[n]

        # Calcula Delta P e Delta Q para cada barra
        n_ref = 0
        for n in range(nbar):
            delta_p[n] = pf[n] - p_calc[n, 0]
            delta_q[n] = qf[n] - q_calc[n, 0]
            if ref[n]:
                if (dger is None) and (f_esp is None):
                    delta_p[n] = 0.0
                delta_q[n] = 0.0
                if (dger is not None) and (f_esp is not None):
                    delta_y[n_ref] = td_esp[n] - td[n]
                    n_ref += 1
            if verifica_pv[n]:
                delta_q[n] = 0.0
            if (dger is not None) and (f_esp is not None):
                if (n+1) in nbss:
                    delta_p[n] = pf[n] + pbss - p_calc[n, 0]

        if (dger is not None) and (f_esp is not None):
            for ng in range(nbarg):
                # barra = int(ng[0]) - 1
                # if ng == 0 or ng == 1:
                #     delta_yi[ng] = 0
                # else:
                if dbar_g[ng, 0] not in (list_renovaveis):
                    delta_yi[ng] = -((pgi[ng] - pg_esp[ng]) + (1/r_est[ng] * (float(f_calc)-f_esp)))

            if warning_pbss:
                delta_ybat = 0
            else:
                delta_ybat = f_esp - f_calc
            delta_pq_yi_y_ylin = np.vstack([delta_p, delta_q, delta_yi, delta_y, delta_ybat])
            # delta_pq_yi_y = np.vstack([delta_p, delta_q, delta_yi, delta_y])
            delta = delta_pq_yi_y_ylin
        else:
            delta = np.vstack([delta_p, delta_q])
            delta_pq = delta
        
        erro_list.append(np.max(np.abs(delta)))

        # Verifica se o algoritimo convergiu
        if np.all(np.abs(delta) < tol):
            convergiu = True
            break

        # Sem a convergência monta-se as submatrizes H, N, M, L
        H_jac = np.zeros([nbar, nbar])
        N_jac = np.zeros([nbar, nbar])
        M_jac = np.zeros([nbar, nbar])
        L_jac = np.zeros([nbar, nbar])
        # Elementos fora da diagonal:
        i=0
        for nlin, linha in enumerate(dlin):
            k = int(nfrom[i] - 1) # "de"
            m = int(nto[i] - 1) # "para"
            # Theta
            tkm = td[k] - td[m]
            tmk = -tkm
            # Condutância
            Gkm = G[k, m]
            # Susceptância
            Bkm = B[k, m]

            # Calcula as submatrizes H, M, N, L
            H_jac[k, m] = (vd[k] * vd[m]) * ((Gkm * sin(tkm)) - (Bkm * cos(tkm)))
            N_jac[k, m] = vd[k] * ((Gkm * cos(tkm)) + (Bkm * sin(tkm)))
            M_jac[k, m] = ((-vd[k]) * vd[m]) * ((Gkm * cos(tkm)) + (Bkm * sin(tkm)))
            L_jac[k, m] = vd[k] * ((Gkm * sin(tkm)) - (Bkm * cos(tkm)))

            H_jac[m, k] = (vd[m] * vd[k]) * ((Gkm * sin(tmk)) - (Bkm * cos(tmk)))
            N_jac[m, k] = vd[m] * ((Gkm * cos(tmk)) + (Bkm * sin(tmk)))
            M_jac[m, k] = ((-vd[m]) * vd[k]) * ((Gkm * cos(tmk)) + (Bkm * sin(tmk)))
            L_jac[m, k] = vd[m] * ((Gkm * sin(tmk)) - (Bkm * cos(tmk)))
            i+=1
        if (dger is not None) and (f_esp is not None):
            A_jac = np.zeros([nbar, nbarg])
            B_jac = np.zeros([nbar, nref])
            C_jac = np.zeros([nbar, nbarg])
            D_jac = np.zeros([nbar, nref])

            BSS_P_JAC = np.zeros([nbar, n_bat])
            BSS_Q_JAC = np.zeros([nbar, n_bat])

            F_jac = np.zeros([nbarg, nbar])
            G_jac = np.zeros([nbarg, nbar])

            J_jac = np.zeros([nbarg, nbarg])
            K_jac = np.zeros([nbarg, nref])
            BSS_YI_JAC = np.zeros([nbarg, n_bat])

            R_jac = np.zeros([nref, nbar])
            S_jac = np.zeros([nref, nbar])
            Y_teta_JAC = np.zeros([n_bat, nbar])
            Y_v_JAC = np.zeros([n_bat, nbar])


            T_jac = np.zeros([nref, nbarg])
            V_jac = np.zeros([nref, nref])
            BSS_Y_JAC = np.zeros([nref, n_bat])
            Y_pgi_JAC = np.zeros([n_bat, nbarg])
            Y_f_JAC = np.ones([n_bat, nref])
            Y_pbss_JAC = np.zeros([n_bat, n_bat])

            for barra in dbar:
                n = int(barra[0]) -1
                for ng in range(nbarg):
                    if n == ng:
                        A_jac[n, ng] = -1

            for barra in dbar:
                n = int(barra[0]) -1
                if (n+1 in nbss):
                    BSS_P_JAC[n, 0] = -1

            for ng in range(nbarg):
                for barra in dbar:
                    n = int(barra[0]) -1

            for ng in range(nbarg):
                for ng2 in range(nbarg):
                    if ng == ng2:
                        J_jac[ng, ng2] = 1
                        # if (ng == 0 and ng2 == 0) or (ng == 1 and ng2 == 1):
                        #     J_jac[ng, ng2] = big_num

            for ng in range(nbarg):
                if dbar_g[ng, 0] not in (list_renovaveis):
                    K_jac[ng, 0] = 1/r_est[ng]

            for barra in dbar:
                n = int(barra[0]) -1
                if ref[n]:
                    R_jac[0, n] = 1

            if warning_pbss:
                for n in range(n_bat):
                    Y_pbss_JAC[n, n] = big_num

        # Elementos da diagonal:
        for n in range(nbar):
            H_jac[n, n] = ((-vd[n]**2) * B[n, n]) - q_calc[n]
            N_jac[n, n] = (p_calc[n] + (vd[n]**2 * G[n, n]))/(vd[n])
            M_jac[n, n] = (p_calc[n]) - (vd[n]**2 * G[n, n])
            L_jac[n, n] = (q_calc[n] - (vd[n]**2 * B[n, n]))/(vd[n])

            if ref[n]:
                if (dger is None) and (f_esp is None):
                    H_jac[n, n] = big_num
                L_jac[n, n] = big_num
            if verifica_pv[n]:
                L_jac[n, n] = big_num

        # Monta a matriz Jacobiana
        JAC = np.vstack([np.hstack([H_jac, N_jac]), np.hstack([M_jac, L_jac])])

        if (dger is not None) and (f_esp is not None):
            JAC_new = np.vstack([np.hstack([H_jac, N_jac, A_jac, B_jac, BSS_P_JAC]), 
                                np.hstack([M_jac, L_jac, C_jac, D_jac, BSS_Q_JAC]),
                                np.hstack([F_jac, G_jac, J_jac, K_jac, BSS_YI_JAC]),
                                np.hstack([R_jac, S_jac, T_jac, V_jac, BSS_Y_JAC]),
                                np.hstack([Y_teta_JAC, Y_v_JAC, Y_pgi_JAC, Y_f_JAC, Y_pbss_JAC])])
            df = pd.DataFrame(JAC_new)
            inv_JAC_new = np.linalg.inv(JAC_new)
            delta_v_teta_pgi_f = np.dot(inv_JAC_new, delta_pq_yi_y_ylin)

            nbar2 = int(2 * nbar)
            nbar3 = int(nbar2 + nbarg)
            nf = nbar3 + 1
            td = td + delta_v_teta_pgi_f[:nbar,0]
            vd = vd + delta_v_teta_pgi_f[nbar:nbar2,0]
            pgi = pgi + delta_v_teta_pgi_f[nbar2:nbar3,0]
            f_calc = f_calc + delta_v_teta_pgi_f[nbar3:nf,0]
            pbss = pbss + delta_v_teta_pgi_f[nf:,0]
            if PbssMax is None:
                PbssMax = 0
            if PbssMin is None:
                PbssMin = 0
            if ((pbss > PbssMax) and (f_calc <= f_esp)):
                pbss = PbssMax
                warning_pbss = True
            if ((pbss > PbssMax) and (f_calc > f_esp)):
                warning_pbss = False
            if ((pbss < PbssMin) and (f_calc < f_esp)):
                warning_pbss = False
            if ((pbss < PbssMin) and (f_calc >= f_esp)):
                pbss = PbssMin
                warning_pbss = True
        else:
            # Inverte a Jacobiana
            inv_JAC = np.linalg.inv(JAC)
            # Resolve sistema de equacoes e calcula DeltaV e DeltaTheta
            delta_v_teta = np.dot(inv_JAC, delta_pq)
            # Atualiza Magnitude e Angula da Tensão
            td = td + delta_v_teta[:nbar,0]
            vd = vd + delta_v_teta[nbar:,0]
        iter += 1

    # Potência reativa gerada nas barras (Qgi)
    qgi = np.zeros(nbar)
    for n in range(nbar):
        qgi[n] = q_calc[n,0] + dbar[n][6]   # Qgi = Qinj + Qload

    end = time.time()

    if (dger is not None) and (f_esp is not None):
        return convergiu, iter, td, vd, erro_list, pgi, qgi, f_calc, pbss, end - start
    else:
        return convergiu, iter, td, vd, erro_list


def get_non_linear_result(path_file: str, pbss_capacity: float, load_multiplier: float) -> pd.DataFrame:
    """
    Função que calcula o flow não linear e retorna o datframe com resultados

    Args:
        path_file (str): Diretorio do caminho do arquivo pwf
        pbss_capacity (float): Capacidade do banco em PU
        load_multiplier (float): multiplicador da carga

    Returns:
        pd.DataFrame: Dataframe com resultados
    """

    with open(path_file, 'r') as f:
        linhas = f.readlines()

    resultados = {}
    titulos = []
    list_titus = ['DBAR', 'DLIN', 'DCER', 'DCSC', 'DCAR', 'DELO', 'DCBA', 'DCLI', 'DCNV', 'DCCV', 'DCCA', 'DCGS', 'DCGE', 'DGER']

    for linha in linhas:
        if any(titus in linha for titus in list_titus):
            titulos.append(linha.replace('\n', ''))

    for titulo in titulos:
        function = f'convert_{titulo}'
        try:
            # Obtém a função pelo nome e chama
            # func = getattr(read_pwf, function)
            resultados[titulo] = globals()[function](linhas) 
        except AttributeError:
            print(f"Função {function} não encontrada.")

    DBAR = resultados['DBAR']
    DBAR = np.array([[d['numero'], d['tipo'], d['tensao'], d['Pg'], d['Qg'], d['Pl'], d['Ql'], d['Sh'], d['angulo']] for d in DBAR])
    colunas = ["numero", "tipo", "tensao", "Pg", "Qg", "Pl", "Ql", "Sh", "angulo"]
    DF_DBAR = pd.DataFrame(DBAR, columns=colunas)

    DLIN = resultados['DLIN']
    DLIN = np.array([[d['De'], d['Para'], d['R'], d['X'], d['Mvar'], d['Tap']] for d in DLIN])
    colunas = ["De", "Para", "R", "X", "Mvar", "Tap"]
    DF_DLIN = pd.DataFrame(DLIN, columns=colunas)

    nbar = DBAR.shape[0]
    nlin = DLIN.shape[0]
    nbarg = np.sum((DBAR[:, 1] == 1) | (DBAR[:, 1] == 2))

    tol = 1e-8
    max_iter = 50

    filtro_geradoras = (DBAR[:, 1] == 1) | (DBAR[:, 1] == 2)
    DBAR_G = DBAR[filtro_geradoras]
    nogen = [int(barra[0]) for barra in DBAR_G]
    nbarg = np.sum((DBAR[:, 1] == 1) | (DBAR[:, 1] == 2))


    DGER = resultados['DGER']
    DGER = np.array([[d['Pmx'], d['Est']] for d in DGER])
    colunas = ['Pmx', 'Est']
    DF_DGER = pd.DataFrame(DGER, columns=colunas)
    f_esp = 1
    f_calc = f_esp

    n_saude = 0.2*pbss_capacity
    n_bateria = pbss_capacity
    if n_bateria > n_saude:
        Pmax = n_bateria - n_saude
    else:
        Pmax = 0
    Pmin = -(pbss_capacity - n_bateria)
    nbss = [3]

    DCCA = resultados['DCCA']
    columns = ['tp', 'num', 'P', 'Q', 'Pt']
    DCCA = pd.DataFrame(np.array([[d['tp'], d['num'], d['P'], d['Q'], d['Pt']] for d in DCCA]), columns=columns)
    DBAR = DBAR.astype(float)
    # MULTIPLICA A CARGA POR UM FATOR MENOR QUE 1 PARA DIMINUIR O VALOR NOMINAL
    DCCA['P'] = DCCA['P'].astype(float)*0.75
    DCCA['Q'] = DCCA['Q'].astype(float)*0.75
    indice = 0
    list_iter = []
    list_td = []
    list_vd = []
    list_max_erro = []
    list_pgi = []
    list_qgi = []
    list_fcalc = []
    list_pbss = []
    list_fcalc_completo = []
    list_pbss_completo = []
    list_n_bateria = []

    # VERFICAÇÃO DE DADOS DE CURVA DE GERAÇÃO SOLAR E EÓLICA
    list_renovaveis = []

    DCGS = resultados['DCGS']
    columns = ['tp', 'num', 'P', 'Q', 'Pt']
    DCGS = pd.DataFrame(np.array([[d['tp'], d['num'], d['P'], d['Q'], d['Pt']] for d in DCGS]), columns=columns)
    DCGS['P'] = DCGS['P'].astype(float)
    DCGS['Q'] = DCGS['Q'].astype(float)
    tipo = str(DCGS['tp'][0])
    if tipo == 'BARR':
        num_solar = int(DCGS['num'][0])
        list_renovaveis.append(num_solar)

    DCGE = resultados['DCGE']
    columns = ['tp', 'num', 'P', 'Q', 'Pt']
    DCGE = pd.DataFrame(np.array([[d['tp'], d['num'], d['P'], d['Q'], d['Pt']] for d in DCGE]), columns=columns)
    DCGE['P'] = DCGE['P'].astype(float)
    DCGE['Q'] = DCGE['Q'].astype(float)
    tipo = str(DCGE['tp'][0])

    if tipo == 'BARR':
        num_eolica = int(DCGE['num'][0])
        list_renovaveis.append(num_eolica)

    times = []

    convergiu = False
    for index, row in DCCA.iterrows():
        DBAR_new = DBAR.copy()
        for i in range(len(DBAR_new)):
            DBAR_new[i][5] *= row['P']
            DBAR_new[i][6] *= row['Q']
        if DCGS is not None:
            row_dcgs = DCGS.loc[index]
            DBAR_new[num_solar-1][3] *= row_dcgs['P']
            # DBAR_new[num_solar][4] *= row_dcgs['Q']
        if DCGE is not None:
            row_dcge = DCGE.loc[index]
            DBAR_new[num_eolica-1][3] *= row_dcge['P']
            # DBAR_new[num_eolica][4] *= row_dcge['Q']
        
        if index > 0 and convergiu:
            # ===============================
            # ATUALIZA DBAR PARA PRÓXIMO PONTO
            # ===============================
            for i in range(nbar):
                DBAR_new[i][2] = vd[i]
                DBAR_new[i][8] = td[i]
            # for k, barra in enumerate(nogen):
            #     DBAR_new[barra-1][3] = pgi[k]

        if row['P'] > 1:
            tipo_carga = 'Pesada'
        elif row['P'] < 1:
            tipo_carga = 'Leve'
        else:
            tipo_carga = 'Nominal'

        # print("\n=== Dados do sistema na última iteração ===")
        # print(f"Tipo de Carga..............: {tipo_carga}")
        # print(f"Frequência Calculada.......: {f_calc*60} Hz")
        # print(f"Potência disponível para carregamento...: {Pmin} pu")
        # print(f"Potência disponível para descarregamento: {Pmax} pu")
        # print(f"Nível da Bateria...........: {(n_bateria/pbss_capacity)*100}%\n")

        Pmax_final = Pmax
        Pmin_final = Pmin

        convergiu, iter, td, vd, erro_list, pgi, qgi, f_calc, pbss, time = solver_newton(DBAR_new, nbar, DLIN, nlin, tol, max_iter, DGER, f_esp, nbss, Pmax_final, Pmin_final, list_renovaveis)
        times.append(time)
        
        if convergiu:
            list_iter.append(iter)
            for i in range(nbar):
                td[i] = round(td[i] * 180 / np.pi, 1)
            list_td.append(td)
            list_vd.append(vd)
            list_max_erro.append(max(erro_list, key=abs))
            list_pgi.append(pgi)
            list_qgi.append(qgi)
            list_fcalc.append(round(float(f_calc.item() * 60), 3))
            list_pbss.append(round(float(pbss), 3))
            n_bateria += -(list_pbss[-1])
            list_n_bateria.append(round(float((n_bateria/pbss_capacity)*100), 0))
            if n_bateria >= pbss_capacity:
                n_bateria = pbss_capacity
                Pmax = n_bateria
                Pmin = 0
            if n_bateria < pbss_capacity:
                if n_bateria > n_saude:
                    Pmax = n_bateria - n_saude
                else:
                    Pmax = 0
                Pmin = -(pbss_capacity - n_bateria)
            indice += 1
        else:
            print('Processo não convergiu')
            list_iter.append(None)
            list_td.append([None]*nbar)
            list_vd.append([None]*nbar)
            list_max_erro.append(None)
            list_pgi.append([None]*nbarg)
            list_fcalc.append(None)
            list_pbss.append(None)
            list_n_bateria.append(None)

            indice += 1

    print(f"Media de tempo Não linear {np.mean(times)}")

    if len(list_iter)>0 and len(list_td)>0 and len(list_vd)>0 and len(list_max_erro)>0 and \
        len(list_pgi)>0 and len(list_fcalc)>0 and len(list_pbss)>0 and len(list_n_bateria)>0 and indice>0:
        num_barras = len(DBAR)
        # Definindo o eixo x
        # indice = len(list_iter)
        x_nr = np.arange(indice)
        FREQ = True

        # CRIAÇÃO DATAFRAME
        data = {
            'Ponto da Curva de Carga': x_nr,
            'Fator de Carga': DCCA['P'],
            'Fator Geração Eólica': DCGE['P'],
            'Fator Geração Solar': DCGS['P'],
            'Iteracoes': list_iter,
            'Frequencia (Hz)': list_fcalc,
            'PBSS': list_pbss,
            'Nivel Bateria (%)': list_n_bateria,
        }
        for i in range(nbar):
            data[f'TD {i+1}'] = [td[i] for td in list_td]

        for i in range(nbar):
            data[f'VD {i+1}'] = [vd[i] for vd in list_vd]

        pgi_labels = [f'Barra-{i}' for i in nogen]
        for i in range(nbarg):
            data[f'PG {pgi_labels[i]}'] = [pgi[i] for pgi in list_pgi]

        qgi_labels = [f'Barra-{i}' for i in nogen]
        for i in range(nbarg):
            data[f'QG {qgi_labels[i]}'] = [qgi[i] for qgi in list_qgi]

        df_resultados = pd.DataFrame(data)

        return df_resultados


# --------------------- FUNCOES AUXILIARES ---------------------- #
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


def txt_to_float(valor):
    try:
        valor_fim = float(valor)
        return valor_fim
    except ValueError as err:
        # print(f'Erro: {err}')
        return 0.0


def txt_to_int(valor):
    try:
        valor_fim = int(valor)
        return valor_fim
    except ValueError as err:
        # print(f'Erro: {err}')
        return 0.0


def txt_to_tensao(valor):
    if valor.find('.') == -1:
        valor_pu = txt_to_float(valor)/1000
        return valor_pu
    else:
        return txt_to_float(valor)

   
def txt_to_tap(valor):
    if valor.find('.') == -1:
        valor_pu = txt_to_float(valor)/1000
        return valor_pu
    else:
        return txt_to_float(valor)


def txt_to_impedancia(valor):
    valor_pu = txt_to_float(valor)/100
    return valor_pu


def txt_to_power(valor):
    valor_pu = txt_to_float(valor)/100
    return valor_pu


def txt_to_load(valor):
    valor_pu = txt_to_float(valor)/100
    return valor_pu


def txt_to_sh(valor):
    valor_pu = txt_to_float(valor)/100
    return valor_pu


def txt_to_mvar(valor):
    valor_pu = txt_to_float(valor)/100
    return valor_pu
    

def start_finish(titu, linhas):
    verificacao = False
    for k,linha in enumerate(linhas):
        if linha[0] != '(':
            if titu in linha:
                linha_ini = k
                verificacao = True
            if  verificacao and linha[0:5] == '99999':
                linha_fim = k
                break
    if verificacao:
        return (linha_ini, linha_fim)
    else:
        return (0, 0)


def convert_DBAR(linhas):
    inicio, final = start_finish('DBAR',linhas)
    barras = []
    for k,line in enumerate(linhas[inicio+1:final]):       
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(No )OETGb(   nome   )Gl( V)( A)( Pg)( Qg)( Qn)( Qm)(Bc  )( Pl)( Ql)( Sh)Are(Vf) 118 barras
            #(Num)OETGb(   nome   )Gl( V)( A)( Pg)( Qg)( Qn)( Qm)(Bc  )( Pl)( Ql)( Sh)Are(Vf) 14 e 6 barras
            d = dict()        
            d['numero'] = txt_to_int(line[0:5])
            d['tipo'] = txt_to_int(line[7])
            d['nome'] = line[10:21]
            # d['tensao'] = txt_to_tensao(line[22:28])
            d['tensao'] = txt_to_tensao(line[24:28])
            d['angulo'] = txt_to_float(line[28:32])
            d['Pg'] = txt_to_power(line[32:37]) #Geração Ativa
            d['Qg'] = txt_to_power(line[37:42]) #Geração Reativa
            d['Qn'] = txt_to_power(line[42:47]) #Geração Reativa Mínima
            d['Qm'] = txt_to_power(line[47:52]) #Geração Reativa Máxima
            d['Bc'] = txt_to_float(line[52:58]) #Barra Controlada
            d['Pl'] = txt_to_load(line[58:63]) #Carga Ativa #carga ativa
            d['Ql'] = txt_to_load(line[63:68]) #Carga Reativa    
            d['Sh'] = txt_to_sh(line[68:73]) #Capacitor ou Reator
            d['Are'] = txt_to_sh(line[73:76]) #Área da Barra
            d['tensao_zip'] = txt_to_tensao(line[76:80]) #Tensão de referência ZIP 
            d['conectividade'] = 0 #Inicialização da conectividade das barras
            barras.append(d)
    return barras


def convert_DLIN(linhas):
    inicio, final = start_finish('DLIN',linhas)
    ramos = []
    indice = 0
    for k,line in enumerate(linhas[inicio+1:final]): 
        if line[0] == '(':
            indice += 1
            
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(De )d O d(Pa )NcEP ( R% )( X% )(Mvar)(Tap)(Tmn)(Tmx)(Phs)(Bc  )(Cn)(Ce)Ns
            d = dict()
            d['De'] = txt_to_int(line[0:5])
            d['Para'] = txt_to_int(line[10:15])
            d['Nc'] = txt_to_int(line[15:17])
            d['R'] = txt_to_impedancia(line[20:26]) #Resistência para linha ou se for trafo para o tap nominal
            d['X'] = txt_to_impedancia(line[26:32]) #Reatância para linha ou se for trafo para o tap nominal
            d['Mvar'] = txt_to_mvar(line[32:38]) #Susceptância shunt
            d['Tap'] = txt_to_tap(line[38:43])
            d['Phs'] = txt_to_float(line[53:58]) #Angulo de defasagem
            ramos.append(d)
    return ramos


def convert_DCCA(linhas):
    inicio, final = start_finish('DCCA',linhas)
    curva = []
    for k,line in enumerate(linhas[inicio+1:final]):
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(tp) (num) C (tp) (num) C (tp) (num) C (tp) (num) O ( P ) ( Q ) (Pt)
            d = dict()
            d['tp'] = line[0:4]
            d['num'] = txt_to_int(line[5:10])
            d['P'] = txt_to_float(line[52:57])
            d['Q'] = txt_to_float(line[58:63])
            d['Pt'] = txt_to_int(line[64:68])
            curva.append(d)
    return curva


def convert_DCGS(linhas):
    inicio, final = start_finish('DCGS',linhas)
    curva = []
    for k,line in enumerate(linhas[inicio+1:final]):
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(tp) (num) C (tp) (num) C (tp) (num) C (tp) (num) O ( P ) ( Q ) (Pt)
            d = dict()
            d['tp'] = line[0:4]
            d['num'] = txt_to_int(line[5:10])
            d['P'] = txt_to_float(line[52:57])
            d['Q'] = txt_to_float(line[58:63])
            d['Pt'] = txt_to_int(line[64:68])
            curva.append(d)
    return curva


def convert_DCGE(linhas):
    inicio, final = start_finish('DCGE',linhas)
    curva = []
    for k,line in enumerate(linhas[inicio+1:final]):
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(tp) (num) C (tp) (num) C (tp) (num) C (tp) (num) O ( P ) ( Q ) (Pt)
            d = dict()
            d['tp'] = line[0:4]
            d['num'] = txt_to_int(line[5:10])
            d['P'] = txt_to_float(line[52:57])
            d['Q'] = txt_to_float(line[58:63])
            d['Pt'] = txt_to_int(line[64:68])
            curva.append(d)
    return curva


def convert_DGER(linhas):
    inicio, final = start_finish('DGER',linhas)
    curva = []
    for k,line in enumerate(linhas[inicio+1:final]):
        if line[0] != '(':
            #0         1         2         3         4         5         6         7          
            #01234567890123456789012345678901234567890123456789012345678901234567890123456789
            #(No ) O (Pmn ) (Pmx ) ( Fp) (FpR) (FPn) (Fa) (Fr) (Ag) ( Xq) (Sno) (Est)
            d = dict()
            d['Pmx'] = txt_to_power(line[15:21])
            d['Est'] = txt_to_impedancia(line[67:72])
            curva.append(d)
    return curva


def mean_relative_error(df_ref, df_test, col, eps=1e-12):
    """
    Calcula o erro médio relativo entre dois dataframes.
    
    df_ref  -> dataframe de referência (denominador)
    df_test -> dataframe comparado
    col     -> nome da coluna comum
    eps     -> evita divisão por zero
    """

    # garantir alinhamento por índice
    s_ref = df_ref[col].values
    s_test = df_test[col].values

    # erro relativo linha a linha
    rel_error = np.abs((s_test - s_ref) / (s_ref + eps))

    # média
    return 100*np.mean(rel_error)


def plot_to_html(fig):
    return fig.to_html(full_html=False)


def plot_serie_pb(df, x_col, y_col, title, y_label, name=None, dash='solid', marker='circle'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        name=name or y_col,
        line=dict(color='black', dash=dash),
        marker=dict(symbol=marker)
    ))
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'y': 0.9, 'xanchor': 'center',
               'font': {'size': 25, 'color': 'black', 'family': 'Arial Black'}},
        xaxis=dict(
            title="Ponto da Curva de Carga",
            title_font=dict(size=20, color='black', family='Arial Black'),
            gridcolor='lightgrey'  # deixa a grade mais discreta
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=20, color='black', family='Arial Black'),
            gridcolor='lightgrey'  # deixa a grade mais discreta
        ),
        height=600,
        legend=dict(font=dict(size=15, family='Arial Black')),
        plot_bgcolor='white',   # fundo da área do gráfico
        paper_bgcolor='white'   # fundo externo
    )
    return fig


def plot_mult_series_pb(df, x_col, cols, title, y_label):
    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    markers = ['circle', 'square', 'diamond', 'cross', 'triangle-up']

    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(
                color='black',
                dash=line_styles[i % len(line_styles)]
            ),
            marker=dict(
                symbol=markers[i % len(markers)]
            )
        ))
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'y': 0.9, 'xanchor': 'center',
               'font': {'size': 25, 'color': 'black', 'family': 'Arial Black'}},
        xaxis=dict(
            title="Ponto da Curva de Carga",
            title_font=dict(size=20, color='black', family='Arial Black'),
            gridcolor='lightgrey'  # deixa a grade mais discreta
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=20, color='black', family='Arial Black'),
            gridcolor='lightgrey'  # deixa a grade mais discreta
        ),
        height=600,
        legend=dict(font=dict(size=15, family='Arial Black')),
        plot_bgcolor='white',   # fundo da área do gráfico
        paper_bgcolor='white'   # fundo externo
    )
    return fig


def save_html(html_parts, filename='NR-Curva-Carga-Baterias.html'):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    {''.join(html_parts)}
    </body>
    </html>
    """

    # Caminho da pasta results dentro do projeto
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Caminho completo do arquivo
    file_path = os.path.join(results_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    webbrowser.open(f'file://{os.path.abspath(file_path)}')


def generate_graph(df, filename: str):
    html_parts = []
    # Gráfico 1.1 - Fator de Carga
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Fator de Carga',
                "Fatores de Carga",
                "Fator de Carga"
            )
        )
    )
    # Gráfico 1.2 - Fator Geração Eólica
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Fator Geração Eólica',
                "Fatores de Geração Eólica",
                "Fator de Geração"
            )
        )
    )
    # Gráfico 1.3 - Fator Geração Solar
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Fator Geração Solar',
                "Fatores de Geração Solar",
                "Fator de Geração"
            )
        )
    )
    # Gráfico 2 - Iterações Método Newton
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Iteracoes',
                "Números de Iterações",
                "Iterações Método Newton"
            )
        )
    )
    # Gráfico 3 - Tensão ANG (TD)
    td_cols = [c for c in df.columns if c.startswith('TD')]
    html_parts.append(
        plot_to_html(
            plot_mult_series_pb(
                df,
                'Ponto da Curva de Carga',
                td_cols,
                "Ângulo da Tensão",
                "Tensão ANG"
            )
        )
    )
    # Gráfico 4 - Tensão MOD (VD)
    vd_cols = [c for c in df.columns if c.startswith('VD')]
    html_parts.append(
        plot_to_html(
            plot_mult_series_pb(
                df,
                'Ponto da Curva de Carga',
                vd_cols,
                "Módulo da Tensão",
                "Tensão MOD"
            )
        )
    )
    # # Gráfico 5 - Máximo Erro
    # html_parts.append(
    #     plot_to_html(
    #         plot_serie_pb(
    #             df,
    #             'Ponto da Curva de Carga',
    #             'Erro Maximo',
    #             "Máximo erro do Método de Newton",
    #             "Máximo Erro",
    #             dash='dot'
    #         )
    #     )
    # )
    
    # Gráfico - PGI
    pgi_cols = [c for c in df.columns if c.startswith('PG Barra')]
    html_parts.append(
        plot_to_html(
            plot_mult_series_pb(
                df,
                'Ponto da Curva de Carga',
                pgi_cols,
                "Potência Gerada por Máquina",
                "Potência Gerada"
            )
        )
    )
    # Gráfico - Frequência
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Frequencia (Hz)',
                "Frequência Calculada",
                "Frequência",
            )
        )
    )
    # Gráfico - Pbss
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'PBSS',
                "Potência Banco de Baterias",
                "Pbss",
            )
        )
    )
    # Gráfico - Nível Baterias
    html_parts.append(
        plot_to_html(
            plot_serie_pb(
                df,
                'Ponto da Curva de Carga',
                'Nivel Bateria (%)',
                "Nível do Banco de Baterias (%)",
                "Nível (%)"
            )
        )
    )
    save_html(html_parts, filename)


def extract_bus_number(col_name):
    # procura padrões: Barra 1, Barra-1, TD 1, VD 1, PG Barra 1, etc.
    match = re.search(r'(?:Barra[-\s]*|TD\s+|VD\s+)(\d+)', col_name)
    if match:
        return match.group(1)
    return "?"


def apply_mono_layout(fig, title, ylabel):
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=26, family="Arial Black", color="black")
        ),

        xaxis=dict(
            title="Time of week",
            title_font=dict(size=18, family="Arial Black"),
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="lightgrey",
            tickfont=dict(size=14, color="black")
        ),

        yaxis=dict(
            title=ylabel,
            title_font=dict(size=18, family="Arial Black"),
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="lightgrey",
            tickfont=dict(size=14, color="black")
        ),

        legend=dict(font=dict(size=24, family="Arial Black")),

        plot_bgcolor="white",
        paper_bgcolor="white",

        height=750,
        margin=dict(l=80, r=50, t=80, b=60)
    )


def generate_graph_multi(dfs: dict, filename: str):
    html_parts = []

    # estilos monocromáticos
    method_markers = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down']
    bus_line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    # ===============================
    # FUNÇÃO BASE PARA GRÁFICOS SIMPLES
    def plot_simple(col, title, ylabel):
        legend_map = {
            "Linear": "Linear Power Flow",
            "Linear com perdas": "Linear (loss estimation)",
            "Não Linear": "Nonlinear Power Flow",
            "Não Linear Sem Bateria": "Nonlinear (No BESS)"
        }

        fig = go.Figure()
        for i, (name, df) in enumerate(dfs.items()):
            legend_name = legend_map.get(name, name)

            fig.add_trace(go.Scatter(
                x=df['Ponto da Curva de Carga'],
                y=df[col] if col != "Fator de Carga" else df[col]*(1/0.75),
                mode='lines+markers',
                name=legend_name,
                line=dict(color='black', dash=bus_line_styles[i % len(bus_line_styles)]),
                marker=dict(symbol=method_markers[i % len(method_markers)])
            ))

        apply_mono_layout(fig, title, ylabel)
        return fig

    # ===============================
    # FUNÇÃO MULTI COLUNAS (TD*, VD*, PG*, QG*)
    def plot_multi_cols_with_dropdown(prefix, title, ylabel):
        method_map = {
            "Linear": "Linear",
            "Linear com perdas": "Linear (loss estimation)",
            "Não Linear": "Nonlinear",
            "Não Linear Sem Bateria": "No BESS"
        }

        fig = go.Figure()
        trace_groups = []

        # mapear barras únicas para estilos de linha
        all_buses = sorted(set(
            extract_bus_number(c)
            for df in dfs.values()
            for c in df.columns if c.startswith(prefix)
        ))
        bus_style_map = {bus: bus_line_styles[i % len(bus_line_styles)] for i, bus in enumerate(all_buses)}

        for df_index, (df_name, df) in enumerate(dfs.items()):
            cols = [c for c in df.columns if c.startswith(prefix)]

            method = method_map.get(df_name, df_name)
            marker_symbol = method_markers[df_index % len(method_markers)]  # dataframe → marker

            for col in cols:
                bus = extract_bus_number(col)
                line_style = bus_style_map.get(bus, 'solid')  # barra → linha

                legend_name = f"{method}" + (f" - Bus {bus}" if bus!= "?" else "")

                fig.add_trace(go.Scatter(
                    x=df['Ponto da Curva de Carga'],
                    y=df[col],
                    mode='lines+markers',
                    name=legend_name,
                    visible=True,

                    # MONOCROMÁTICO
                    line=dict(color='black', dash=line_style),
                    marker=dict(symbol=marker_symbol)
                ))

                trace_groups.append(df_name)

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                x=1.12,
                y=1.2
            )]
        )

        apply_mono_layout(fig, title, ylabel)
        return fig

    # ===============================
    # SEM DROPDOWN (comparação direta)
    html_parts.append(plot_to_html(plot_simple('Fator de Carga', "Load factor", "Factor")))
    html_parts.append(plot_to_html(plot_simple('Fator Geração Eólica', "Wind generation factor", "Factor")))
    html_parts.append(plot_to_html(plot_simple('Fator Geração Solar', "Solar generation factor", "Factor")))
    html_parts.append(plot_to_html(plot_simple('Iteracoes', "Number of Iterations", "Iterations (#)")))

    # ===============================
    # COM DROPDOWN
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown("PG Barra", "Active Power Generation", "P (pu)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown("QG Barra", "Reactive Power Generation", "Q (pu)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown("TD", "Voltage Phase", "Phase (°)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown("VD", "Voltage Magnitude", "V (pu)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown('Frequencia (Hz)', "Frequency", "F (Hz)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown('PBSS', "BESS Active Power", "P (pu)")))
    html_parts.append(plot_to_html(plot_multi_cols_with_dropdown('Nivel Bateria (%)', "BESS Level", " BL (%)")))

    # ===============================
    save_html(html_parts, filename)
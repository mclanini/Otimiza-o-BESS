import os
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs


def plot_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


def salvar_html(html_parts, filename='NR-Curva-Carga-Baterias.html'):
    html_content = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8" />
    <script>{get_plotlyjs()}</script>
</head>
<body>
{''.join(html_parts)}
</body>
</html>'''
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    webbrowser.open(f'file://{os.path.abspath(filename)}')


ESTILOS = {
    'Sem perdas': dict(dash='solid', symbol='circle-open', size=6),
    'Com perdas': dict(dash='dash', symbol='triangle-up', size=6),
}


def plot_comparativo(df, y_col, titulo, y_label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
    series = {}
    for cenario, estilo in ESTILOS.items():
        dados = df.loc[df['Cenário'] == cenario].sort_values('Capacidade Bateria (p.u.)')
        series[cenario] = dados.set_index('Capacidade Bateria (p.u.)')[y_col]
        fig.add_trace(go.Scatter(
            x=dados['Capacidade Bateria (p.u.)'],
            y=dados[y_col],
            mode='lines+markers',
            name=cenario,
            line=dict(color='black', dash=estilo['dash']),
            marker=dict(symbol=estilo['symbol'], size=estilo['size'], line=dict(width=1.5)),
            connectgaps=False,
        ), row=1, col=1)

    dif = series['Com perdas'] - series['Sem perdas']
    fig.add_trace(go.Scatter(
        x=dif.index, y=dif.values,
        mode='lines+markers',
        name='Diferença (Com − Sem perdas)',
        line=dict(color='crimson'),
        marker=dict(symbol='diamond', size=5),
        connectgaps=False,
    ), row=2, col=1)

    fonte = dict(size=17, color='black', family='Arial Black')
    fig.update_xaxes(gridcolor='lightgrey', dtick=5)
    fig.update_xaxes(title='Capacidade Bateria (p.u.)', title_font=fonte, row=2, col=1)
    fig.update_yaxes(title=y_label, title_font=fonte, gridcolor='lightgrey', row=1, col=1)
    fig.update_yaxes(title='Diferença', title_font=fonte, gridcolor='lightgrey',
                     exponentformat='power', row=2, col=1)
    fig.update_layout(
        title=dict(text=titulo, x=0.5, y=0.97, xanchor='center',
                   font=dict(size=24, color='black', family='Arial Black')),
        height=800,
        showlegend=True,
        legend=dict(font=dict(size=15, family='Arial Black'),
                    orientation='h', x=0.5, xanchor='center', y=-0.12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=35, t=110, b=110),
    )
    return fig


def gera_gráficos_curva_new(df):
    colunas = {'Cenário', 'Capacidade Bateria (p.u.)',
               'Corte de Carga', 'Função Objetivo'}
    faltantes = colunas - set(df.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes no DataFrame: {sorted(faltantes)}')

    graficos = [
        (df, 'Corte de Carga',
         'Efeito da Capacidade da Bateria sobre o Corte de Carga',
         'Corte de Carga (MW)'),
        (df, 'Função Objetivo',
         'Efeito da Capacidade da Bateria sobre a Função Objetivo',
         'Função Objetivo (R$)'),
        (df.loc[df['Capacidade Bateria (p.u.)'] >= 68], 'Função Objetivo',
         'Efeito da Capacidade da Bateria sobre a Função Objetivo (a partir de 68)',
         'Função Objetivo (R$)'),
    ]
    html_parts = [
        plot_to_html(plot_comparativo(dados, coluna, titulo, eixo_y))
        for dados, coluna, titulo, eixo_y in graficos
    ]
    salvar_html(html_parts)

# ---------------------------------------------------------------------------
# NOVO: comparação de diferentes barras, sempre com perdas no balanço DC.
# Mantém as funções anteriores de gráficos intactas.
# ---------------------------------------------------------------------------
def gera_graficos_alocacao_bateria(df, filename='comparativo_alocacao_bateria.html',
                                 abrir_navegador=True):
    """Compara, por capacidade, as barras candidatas sem misturar cenários.

    Cada gráfico tem: curvas por barra em cima e diferenças para a primeira
    barra da lista (referência) em baixo. Valores sem solução ficam como NaN.
    """
    import pandas as pd

    exigidas = {'Barra Bateria', 'Capacidade Bateria (p.u.)',
                'Corte de Carga (MW)', 'Função Objetivo',
                'Frequência (Hz)', 'Potência Bateria (MW)', 'Perdas Totais (MW)'}
    faltantes = exigidas - set(df.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes: {sorted(faltantes)}')
    if df.empty:
        raise ValueError('Nenhum resultado fornecido para os gráficos.')

    barras = list(dict.fromkeys(df['Barra Bateria'].astype(int).tolist()))
    referencia = barras[0]
    metricas = [
        ('Corte de Carga (MW)', 'Efeito da Capacidade da Bateria sobre o Corte de Carga', 'Corte de carga (MW)'),
        ('Função Objetivo', 'Efeito da Capacidade da Bateria sobre a Função Objetivo', 'Valor da função objetivo'),
        ('Frequência (Hz)', 'Efeito da Capacidade da Bateria sobre a Frequência', 'Frequência (Hz)'),
        ('Potência Bateria (MW)', 'Efeito da Capacidade da Bateria sobre a Potência Injetada', 'Potência (MW)'),
        ('Perdas Totais (MW)', 'Efeito da Capacidade da Bateria sobre as Perdas Elétricas', 'Perdas (MW)'),
    ]
    cores = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
    html_parts = []
    for coluna, titulo, label_y in metricas:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.69, 0.31], vertical_spacing=0.09)
        base = (df.loc[df['Barra Bateria'] == referencia,
                       ['Capacidade Bateria (p.u.)', coluna]]
                .set_index('Capacidade Bateria (p.u.)')[coluna])

        for idx, barra in enumerate(barras):
            dados = df.loc[df['Barra Bateria'] == barra].sort_values('Capacidade Bateria (p.u.)')
            cor = cores[idx % len(cores)]
            fig.add_trace(go.Scatter(
                x=dados['Capacidade Bateria (p.u.)'], y=dados[coluna],
                mode='lines+markers', name=f'Barra {barra}',
                legendgroup=str(barra), line=dict(color=cor, width=2),
                marker=dict(size=5), connectgaps=False,
                customdata=dados[['Status']].to_numpy() if 'Status' in dados else None,
                hovertemplate='Capacidade: %{x:.3f} p.u.<br>Valor: %{y:.6g}<extra>%{fullData.name}</extra>',
            ), row=1, col=1)
            alinhado = dados.set_index('Capacidade Bateria (p.u.)')[coluna]
            dif = alinhado - base.reindex(alinhado.index)
            fig.add_trace(go.Scatter(
                x=dif.index, y=dif.values, mode='lines+markers',
                name=f'Barra {barra} - Barra {referencia}',
                legendgroup=str(barra), showlegend=False,
                line=dict(color=cor, width=1.7), marker=dict(size=4),
                connectgaps=False,
                hovertemplate='Capacidade: %{x:.3f} p.u.<br>Diferença: %{y:.6g}<extra>%{fullData.name}</extra>',
            ), row=2, col=1)
        fig.add_hline(y=0, line_color='#888', line_dash='dot', row=2, col=1)
        fig.update_xaxes(gridcolor='lightgrey', row=1, col=1)
        fig.update_xaxes(title='Capacidade Bateria (p.u.)', gridcolor='lightgrey', row=2, col=1)
        fig.update_yaxes(title=label_y, gridcolor='lightgrey', row=1, col=1)
        fig.update_yaxes(title=f'Diferença vs barra {referencia}', gridcolor='lightgrey', row=2, col=1)
        fig.update_layout(title=dict(
                              text=titulo, x=0.5, y=0.97, xanchor='center',
                              font=dict(size=26, color='black', family='Arial Black')
                          ),
                          height=740, template='plotly_white',
                          legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.13),
                          margin=dict(l=85, r=30, t=95, b=100))
        html_parts.append(plot_to_html(fig))

    html = ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"/>'
            f'<script>{get_plotlyjs()}</script></head><body>'
            + '\n'.join(html_parts) + '</body></html>')
    with open(filename, 'w', encoding='utf-8') as arq:
        arq.write(html)
    if abrir_navegador:
        webbrowser.open('file://' + os.path.abspath(filename))
    return filename

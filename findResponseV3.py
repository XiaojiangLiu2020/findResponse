import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import os
import numpy as np
import sys
import webbrowser
import threading
import copy
import time

# --- PyInstaller 路径处理 ---
if getattr(sys, 'frozen', False):
    assets_path = os.path.join(sys._MEIPASS, 'assets')
else:
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

# --- 初始化应用 ---
app = dash.Dash(__name__, assets_folder=assets_path, suppress_callback_exceptions=True)
server = app.server

# --- 全局变量 ---
original_data = None
processed_data = None

# --- 自定义 Plotly 模板 ---
custom_template = {
    "layout": go.Layout(
        font={"family": "Segoe UI, sans-serif", "color": "#333"},
        title_font={"size": 20, "color": "#111"},
        legend_title_font_color="#444",
        xaxis={"gridcolor": "#e5e5e5", "zerolinecolor": "#ddd", "linecolor": "#ddd"},
        yaxis={"gridcolor": "#e5e5e5", "zerolinecolor": "#ddd", "linecolor": "#ddd"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
}

# --- 应用布局 ---
app.layout = html.Div(id="app-container", children=[
    # 后台数据存储
    dcc.Store(id='analysis-mode-store', data='single'),
    dcc.Store(id='marking-mode-store', data='none'),
    dcc.Store(id='marked-points-store', data={
        'single': {'peaks': {}, 'intake': {}, 'exhaust': {}},
        'multi': {'peaks': [], 'intake': [], 'exhaust': []}
    }),
    dcc.Store(id='previous-marked-points-store', data=None),
    dcc.Store(id='previous-data-store', data=None),
    dcc.Store(id='last-action-store', data=None),
    dcc.Store(id='log-store', data=''),
    dcc.Store(id='metric-annotations-store', data=[]),
    dcc.Store(id='results-data-store', data=None),

    # 校准相关 Store
    dcc.Store(id='calibration-store', data={'applied': False}),
    dcc.Store(id='baseline-points-store', data=[]),

    # 标题
    html.Div(id="header", children=[html.H1("气体传感器数据分析平台")]),

    html.Div(id="main-content", children=[
        # 左侧控制面板
        html.Div(id="control-panel", children=[
            # 通用控制部分
            html.Div(className="control-card", children=[
                html.H3("1. 上传文件"),
                dcc.Upload(id="upload-data",
                           children=html.Div(id="upload-output", children=["📂 拖放或点击上传 CSV/Excel 文件"]),
                           multiple=False,
                           accept='.csv,.xls,.xlsx'),
            ]),
            html.Div(className="control-card", children=[
                html.H3("2. 数据预处理"),
                html.Div(className="control-group", children=[
                    html.Label("截取数据范围 (行号):"),
                    dcc.Input(id="range-start", type="number", placeholder="起始行号"),
                    dcc.Input(id="range-end", type="number", placeholder="结束行号"),
                    html.Button("截取数据", id="trim-button", n_clicks=0),
                ]),
                html.Div(className="control-group", children=[
                    html.Label("采样频率 (Hz):"),
                    dcc.Input(id="sampling-rate-input", type="number", placeholder="例如: 1", value=1, min=0.0001,
                              step="any", debounce=True),
                ]),
            ]),

            # 包含 分析模式 和 基线校准 的选项卡区域
            html.Div(id='analysis-controls', className="control-card", style={'padding': '0'}, children=[
                dcc.Tabs(id="control-tabs", value='tab-analysis', className="custom-tabs", children=[
                    # Tab 1: 原有的分析设置
                    dcc.Tab(label='分析设置', value='tab-analysis', className="custom-tab",
                            selected_className="custom-tab--selected", children=[
                            html.Div(style={'padding': '20px'}, children=[
                                html.H3("3. 操作与模式设置", style={'margin-top': '0'}),
                                html.Div(className="control-group", children=[
                                    html.Label("分析模式"),
                                    dcc.RadioItems(
                                        id='analysis-mode-selector',
                                        options=[
                                            {'label': ' 单传感器分析', 'value': 'single'},
                                            {'label': ' 多传感器分析', 'value': 'multi'},
                                        ],
                                        value='single',
                                        labelStyle={'display': 'block', 'margin-bottom': '10px', 'cursor': 'pointer'},
                                        inputStyle={'margin-right': '10px'}
                                    ),
                                ]),
                                html.Div(className="control-group", children=[
                                    html.Label("指标字体大小:"),
                                    dcc.Input(id="annotation-font-size-input", type="number", placeholder="例如: 9",
                                              value=9, min=1,
                                              step=1, debounce=True),
                                ]),
                                html.Hr(style={'margin': '20px 0'}),
                                html.Div(className="control-group", children=[
                                    html.Label("文件操作"),
                                    html.Button("下载当前数据 (CSV)", id="download-button", n_clicks=0),
                                ]),
                            ])
                        ]),

                    # Tab 2: 增强的基线校准模式
                    dcc.Tab(label='基线校准', value='tab-calibration', className="custom-tab",
                            selected_className="custom-tab--selected", children=[
                            html.Div(style={'padding': '20px'}, children=[
                                html.H3("基线校准模式", style={'margin-top': '0'}),

                                html.Div(className="control-group", children=[
                                    html.Label("校准算法:"),
                                    dcc.Dropdown(
                                        id='calib-method',
                                        options=[
                                            {'label': '比值法 (R / R0)', 'value': 'div'},
                                            {'label': '差值法 (R - R0)', 'value': 'sub'},
                                            # --- MODIFIED: 新增算法选项 ---
                                            {'label': '反比值法 (1 - R/R0)', 'value': 'one_minus_div'},
                                        ],
                                        value='div',
                                        clearable=False
                                    ),
                                ]),

                                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),

                                # 模式 A: 固定范围
                                html.Label("方式一：固定范围平均", style={'fontWeight': 'bold', 'color': '#555'}),
                                html.Div(className="control-group",
                                         style={'display': 'flex', 'gap': '5px', 'marginBottom': '5px'}, children=[
                                        dcc.Input(id="calib-start", type="number", placeholder="Start", value=0,
                                                  style={'width': '80px'}),
                                        dcc.Input(id="calib-end", type="number", placeholder="End", value=10,
                                                  style={'width': '80px'}),
                                        html.Button("应用固定", id="apply-calib-constant-button", n_clicks=0,
                                                    style={'backgroundColor': '#17a2b8', 'color': 'white', 'flex': '1',
                                                           'padding': '5px'}),
                                    ]),

                                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),

                                # 模式 B: 线性拟合
                                html.Label("方式二：多点线性拟合 (漂移校准)",
                                           style={'fontWeight': 'bold', 'color': '#555'}),
                                html.Div(className="control-group",
                                         style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
                                        html.Button("1. 点击选择基线点", id="btn-select-baseline-points", n_clicks=0,
                                                    style={'backgroundColor': '#6f42c1', 'color': 'white'}),
                                        html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                                            html.Button("2. 拟合并应用", id="apply-calib-linear-button", n_clicks=0,
                                                        style={'backgroundColor': '#28a745', 'color': 'white',
                                                               'flex': '1'}),
                                            html.Button("清除点", id="clear-baseline-points-button", n_clicks=0,
                                                        style={'backgroundColor': '#6c757d', 'color': 'white',
                                                               'flex': '0.5'}),
                                        ])
                                    ]),

                                html.Hr(style={'margin': '20px 0'}),

                                html.Button("重置原始数据", id="reset-calib-button", n_clicks=0,
                                            style={'backgroundColor': '#dc3545', 'color': 'white', 'width': '100%'}),

                                html.Div(id='calib-status',
                                         style={'marginTop': '15px', 'fontSize': '0.85em', 'color': '#007bff',
                                                'whiteSpace': 'pre-wrap'})
                            ])
                        ]),
                ]),
            ]),
        ]),
        # 右侧图表与结果区域
        html.Div(id="graph-container", children=[
            html.Div(className="graph-card", children=[
                html.H3("数据可视化与标记"),
                html.Div(id='single-sensor-controls', className="column-selector-wrapper", children=[
                    dcc.Dropdown(id='column-selector', multi=False, placeholder="选择要显示的列...")
                ]),
                dcc.Graph(id="main-plot"),
                html.Div(className="marking-controls", children=[
                    html.Button("标记峰值", id="mark-peak-button", n_clicks=0, className="btn-peak"),
                    html.Button("标记进气点", id="mark-intake-button", n_clicks=0, className="btn-intake"),
                    html.Button("标记出气点", id="mark-exhaust-button", n_clicks=0, className="btn-exhaust"),
                    html.Button("结束标记", id="end-marking-button", n_clicks=0, className="btn-end"),
                    html.Button("撤销上一步", id="undo-button", n_clicks=0, className="btn-undo"),
                    html.Button("计算指标", id="calculate-button", n_clicks=0, className="btn-calculate"),
                    html.Button("清除标记", id="clear-marks-button", n_clicks=0, className="btn-clear"),
                ])
            ]),
            html.Div(id='results-container', className='results-card')
        ]),
    ]),
    # 下载组件
    dcc.Download(id="download-file"),
    dcc.Download(id="download-results-csv"),
])


# --- 核心函数 ---

def generate_results_table(results_data):
    if not results_data: return None
    headers = ["Sensor", "Peak Index", "Peak Value", "Intake Index", "Intake Value", "Response Size (S)",
               "Rgas/Rair", "Rair/Rgas", "Response Time (t90)", "Recovery Time (RT90)"]
    header_row = html.Tr([html.Th(h) for h in headers])
    body_rows = [html.Tr([html.Td(row_data.get(h, '')) for h in headers]) for row_data in results_data]
    return html.Table([html.Thead(header_row), html.Tbody(body_rows)], className="results-table")


def calculate_metrics(marked_points, data, column, analysis_mode='single', sampling_rate=1):
    log_messages = []
    results_data = []
    annotations = []
    sampling_rate = sampling_rate if sampling_rate is not None and sampling_rate > 0 else 1
    log_messages.append(f"Calculation started with sampling rate: {sampling_rate} Hz")

    if analysis_mode == 'single':
        peaks = sorted(marked_points.get('single', {}).get('peaks', {}).get(column, []), key=lambda p: p[0])
        intakes = sorted(marked_points.get('single', {}).get('intake', {}).get(column, []), key=lambda p: p[0])
        exhausts = sorted(marked_points.get('single', {}).get('exhaust', {}).get(column, []), key=lambda p: p[0])

        if not peaks or not intakes:
            return ([html.P("错误: 请至少标记一个峰值点和一个进气点。")], "", [], None)

        series = data[column]
        for i, (x_peak, y_peak) in enumerate(peaks):
            log_messages.append(f"\n{'=' * 50}\nProcessing Peak #{i + 1} at index={x_peak:.2f}, y={y_peak:.4f}")
            possible_intakes = [p for p in intakes if p[0] < x_peak]
            if not possible_intakes: continue
            x_gasin, y_gasin = max(possible_intakes, key=lambda p: p[0])
            log_messages.append(f"Paired intake point: index={x_gasin:.2f}, y={y_gasin:.4f}")

            r_gas_div_air = y_peak / y_gasin if y_gasin != 0 else float('inf')
            r_air_div_gas = y_gasin / y_peak if y_peak != 0 else float('inf')
            response_size = (y_peak - y_gasin) / y_gasin if y_gasin != 0 else float('inf')
            response_time_val, recovery_time_val = None, None

            try:
                y_target_response = y_gasin + 0.9 * (y_peak - y_gasin)
                for idx in range(int(round(x_peak)), int(round(x_gasin)), -1):
                    if idx > 0 and idx < len(series):
                        if (series.iloc[idx - 1] - y_target_response) * (series.iloc[idx] - y_target_response) <= 0:
                            closest_x = idx if abs(series.iloc[idx] - y_target_response) < abs(
                                series.iloc[idx - 1] - y_target_response) else idx - 1
                            response_time_val = (closest_x - x_gasin) / sampling_rate
                            break
            except Exception as e:
                log_messages.append(f"Exception during response time calculation: {e}")

            if response_time_val is not None:
                annotations.append({'x': x_peak / sampling_rate, 'y': y_peak,
                                    'text': f"Resp: {response_size:.1%}<br>T: {response_time_val:.1f}s",
                                    'column': column})

            possible_exhausts = [p for p in exhausts if p[0] > x_peak]
            if possible_exhausts:
                x_gasout, _ = min(possible_exhausts, key=lambda p: p[0])
                try:
                    y_target_recovery = y_gasin + 0.1 * (y_peak - y_gasin)
                    recovery_slice = series[series.index >= int(round(x_gasout))]
                    for idx in recovery_slice.index:
                        if idx > 0 and idx < len(series):
                            if (series.iloc[idx - 1] - y_target_recovery) * (series.iloc[idx] - y_target_recovery) <= 0:
                                closest_x = idx if abs(series.iloc[idx] - y_target_recovery) < abs(
                                    series.iloc[idx - 1] - y_target_recovery) else idx - 1
                                recovery_time_val = (closest_x - x_gasout) / sampling_rate
                                break
                except Exception as e:
                    log_messages.append(f"Exception during recovery time calculation: {e}")

            result_row = {"Sensor": column, "Peak Index": f"{x_peak:.2f}", "Peak Value": f"{y_peak:.4f}",
                          "Intake Index": f"{x_gasin:.2f}", "Intake Value": f"{y_gasin:.4f}",
                          "Response Size (S)": f"{response_size:.2%}" if response_size != float('inf') else "inf",
                          "Rgas/Rair": f"{r_gas_div_air:.4f}" if r_gas_div_air != float('inf') else "inf",
                          "Rair/Rgas": f"{r_air_div_gas:.4f}" if r_air_div_gas != float('inf') else "inf",
                          "Response Time (t90)": f"{response_time_val:.2f}" if response_time_val is not None else "N/A",
                          "Recovery Time (RT90)": f"{recovery_time_val:.2f}" if recovery_time_val is not None else "N/A"}
            results_data.append(result_row)

    elif analysis_mode == 'multi':
        peaks_x = sorted(marked_points.get('multi', {}).get('peaks', []))
        intakes_x = sorted(marked_points.get('multi', {}).get('intake', []))
        exhausts_x = sorted(marked_points.get('multi', {}).get('exhaust', []))

        if not peaks_x or not intakes_x:
            return ([html.P("错误: 请至少标记一个峰值点和一个进气点。")], "", [], None)

        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        for col_name in numeric_columns:
            series = data[col_name]
            for i, x_peak_idx in enumerate(peaks_x):
                possible_intakes_x = [x for x in intakes_x if x < x_peak_idx]
                if not possible_intakes_x: continue
                x_gasin_idx = max(possible_intakes_x)
                y_peak, y_gasin = series.iloc[int(round(x_peak_idx))], series.iloc[int(round(x_gasin_idx))]

                r_gas_div_air = y_peak / y_gasin if y_gasin != 0 else float('inf')
                r_air_div_gas = y_gasin / y_peak if y_peak != 0 else float('inf')
                response_size = (y_peak - y_gasin) / y_gasin if y_gasin != 0 else float('inf')
                response_time_val, recovery_time_val = None, None

                try:
                    y_target_response = y_gasin + 0.9 * (y_peak - y_gasin)
                    for idx in range(int(round(x_peak_idx)), int(round(x_gasin_idx)), -1):
                        if idx > 0 and (series.iloc[idx - 1] - y_target_response) * (
                                series.iloc[idx] - y_target_response) <= 0:
                            closest_x_idx = idx if abs(series.iloc[idx] - y_target_response) < abs(
                                series.iloc[idx - 1] - y_target_response) else idx - 1
                            response_time_val = (closest_x_idx - x_gasin_idx) / sampling_rate
                            break
                except Exception as e:
                    log_messages.append(f"Exception during response time calculation: {e}")

                if response_time_val is not None:
                    annotations.append({'x': x_peak_idx / sampling_rate, 'y': y_peak,
                                        'text': f"Resp: {response_size:.1%}<br>T: {response_time_val:.1f}s",
                                        'column': col_name})

                possible_exhausts_x = [x for x in exhausts_x if x > x_peak_idx]
                if possible_exhausts_x:
                    x_gasout_idx = min(possible_exhausts_x)
                    try:
                        y_target_recovery = y_gasin + 0.1 * (y_peak - y_gasin)
                        recovery_slice = series[series.index >= x_gasout_idx]
                        for idx in recovery_slice.index:
                            if idx > 0 and (series.iloc[idx - 1] - y_target_recovery) * (
                                    series.iloc[idx] - y_target_recovery) <= 0:
                                closest_x_idx = idx if abs(series.iloc[idx] - y_target_recovery) < abs(
                                    series.iloc[idx - 1] - y_target_recovery) else idx - 1
                                recovery_time_val = (closest_x_idx - x_gasout_idx) / sampling_rate
                                break
                    except Exception as e:
                        log_messages.append(f"Exception during recovery time calculation: {e}")

                result_row = {"Sensor": col_name, "Peak Index": f"{x_peak_idx:.2f}", "Peak Value": f"{y_peak:.4f}",
                              "Intake Index": f"{x_gasin_idx:.2f}", "Intake Value": f"{y_gasin:.4f}",
                              "Response Size (S)": f"{response_size:.2%}" if response_size != float('inf') else "inf",
                              "Rgas/Rair": f"{r_gas_div_air:.4f}" if r_gas_div_air != float('inf') else "inf",
                              "Rair/Rgas": f"{r_air_div_gas:.4f}" if r_air_div_gas != float('inf') else "inf",
                              "Response Time (t90)": f"{response_time_val:.2f}" if response_time_val is not None else "N/A",
                              "Recovery Time (RT90)": f"{recovery_time_val:.2f}" if recovery_time_val is not None else "N/A"}
                results_data.append(result_row)

    if not results_data:
        return ([html.P("无法找到任何有效的'进气-峰值'组合。请检查您的标记。")], "\n".join(log_messages), [], None)
    return results_data, "\n".join(log_messages), annotations


def create_figure(data, title, analysis_mode, selected_column=None, marked_points=None, sampling_rate=1,
                  annotations_data=None, annotation_font_size=9, baseline_points=None):
    fig = go.Figure()
    sampling_rate = sampling_rate if sampling_rate is not None and sampling_rate > 0 else 1

    if data is None or data.empty:
        fig.update_layout(title=title, template=custom_template, xaxis={"visible": False}, yaxis={"visible": False},
                          annotations=[{"text": "No data", "xref": "paper", "yref": "paper", "showarrow": False,
                                        "font": {"size": 16}}])
        return fig

    time_axis = data.index / sampling_rate

    # 绘制主数据线
    if analysis_mode == 'single':
        if selected_column and selected_column in data.columns and pd.api.types.is_numeric_dtype(data[selected_column]):
            fig.add_trace(go.Scatter(x=time_axis, y=data[selected_column], mode="lines", name=selected_column))
        else:
            fig.update_layout(title=title, template=custom_template, annotations=[
                {"text": "Please select a numeric column", "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 16}}])
    elif analysis_mode == 'multi':
        for col in data.select_dtypes(include=np.number).columns:
            fig.add_trace(go.Scatter(x=time_axis, y=data[col], mode="lines", name=col))

    # 绘制用于基线拟合的选点 (紫色虚线)
    if baseline_points:
        for i, x_idx in enumerate(baseline_points):
            fig.add_shape(type="line", x0=x_idx / sampling_rate, y0=0, x1=x_idx / sampling_rate, y1=1, yref="paper",
                          line=dict(color="#6f42c1", width=2, dash="dashdot"),
                          name="Baseline Point" if i == 0 else None,
                          showlegend=False)

    # 绘制标记点 (Peak, Intake, Exhaust)
    if marked_points:
        if analysis_mode == 'single' and selected_column:
            point_types = {'peaks': ('Peak', 'red', 'x-thin'), 'intake': ('Intake', 'green', 'circle-open'),
                           'exhaust': ('Exhaust', 'blue', 'diamond-open')}
            for p_type, (name, color, symbol) in point_types.items():
                points = marked_points.get('single', {}).get(p_type, {}).get(selected_column, [])
                if points:
                    fig.add_trace(
                        go.Scatter(x=[p[0] / sampling_rate for p in points], y=[p[1] for p in points], mode='markers',
                                   name=name, marker=dict(color=color, size=10, symbol=symbol, line={'width': 2})))
        elif analysis_mode == 'multi':
            line_types = {'peaks': ('Peak Line', 'red'), 'intake': ('Intake Line', 'green'),
                          'exhaust': ('Exhaust Line', 'blue')}
            for l_type, (name, color) in line_types.items():
                x_vals = marked_points.get('multi', {}).get(l_type, [])
                for i, x in enumerate(x_vals):
                    fig.add_shape(type="line", x0=x / sampling_rate, y0=0, x1=x / sampling_rate, y1=1, yref="paper",
                                  line=dict(color=color, width=2, dash="solid"), name=name if i == 0 else None,
                                  showlegend=i == 0)

    # 绘制计算结果的标注
    if annotations_data:
        font_size = annotation_font_size if annotation_font_size is not None and annotation_font_size > 0 else 9
        for ann in annotations_data:
            if analysis_mode == 'single' and ann.get('column') != selected_column: continue
            fig.add_annotation(x=ann['x'], y=ann['y'], text=ann['text'], showarrow=True,
                               font=dict(size=font_size, color="black"), align="left", arrowhead=2, arrowsize=1,
                               arrowwidth=1.5, arrowcolor="#636363", ax=20, ay=-40, bordercolor="#c7c7c7",
                               borderwidth=1, borderpad=4, bgcolor="rgba(255, 255, 224, 0.85)", opacity=0.9)

    fig.update_layout(title=title, xaxis_title="Time (seconds)", yaxis_title="Value", legend_title="Legend",
                      template=custom_template, margin=dict(l=40, r=40, t=60, b=40), showlegend=True)
    return fig


# --- 回调函数 ---

@app.callback(Output('single-sensor-controls', 'style'), Input('analysis-mode-selector', 'value'))
def toggle_ui_controls(mode):
    return {'display': 'block'} if mode == 'single' else {'display': 'none'}


@app.callback(Output('analysis-mode-store', 'data'), Input('analysis-mode-selector', 'value'))
def store_analysis_mode(mode): return mode


@app.callback(Output('marking-mode-store', 'data'),
              [Input('mark-peak-button', 'n_clicks'), Input('mark-intake-button', 'n_clicks'),
               Input('mark-exhaust-button', 'n_clicks'), Input('end-marking-button', 'n_clicks'),
               Input('btn-select-baseline-points', 'n_clicks')])
def set_marking_mode(peak_clicks, intake_clicks, exhaust_clicks, end_clicks, baseline_clicks):
    trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    mapping = {
        'mark-peak-button': 'peaks',
        'mark-intake-button': 'intake',
        'mark-exhaust-button': 'exhaust',
        'btn-select-baseline-points': 'baseline'
    }
    return mapping.get(trigger_id, 'none')


# --- MODIFIED: 更新校准状态回调以支持新算法名称 ---
@app.callback(
    [Output('calibration-store', 'data'), Output('calib-status', 'children'),
     Output('baseline-points-store', 'data', allow_duplicate=True)],
    [Input('apply-calib-constant-button', 'n_clicks'),
     Input('apply-calib-linear-button', 'n_clicks'),
     Input('reset-calib-button', 'n_clicks'),
     Input('clear-baseline-points-button', 'n_clicks')],
    [State('calib-start', 'value'), State('calib-end', 'value'),
     State('calib-method', 'value'), State('baseline-points-store', 'data')],
    prevent_initial_call=True
)
def update_calibration_store(btn_constant, btn_linear, btn_reset, btn_clear, start, end, method, baseline_points):
    ctx = callback_context
    if not ctx.triggered: return no_update, no_update, no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-calib-button':
        return {'applied': False}, "状态: 已重置 (无校准)", []

    if trigger_id == 'clear-baseline-points-button':
        return no_update, "状态: 已清除选择点", []

    method_map = {
        'div': '比值法 (R/R0)',
        'sub': '差值法 (R-R0)',
        'one_minus_div': '反比值法 (1 - R/R0)'
    }
    method_name = method_map.get(method, '未知算法')

    if trigger_id == 'apply-calib-constant-button':
        if start is None or end is None or start >= end:
            return no_update, "错误: 起始行必须小于结束行", no_update
        return {'applied': True, 'type': 'constant', 'range': [start, end], 'method': method}, \
            f"状态: 已应用 [固定范围] 校准\n算法: {method_name}\n范围: 行 {start} - {end}", no_update

    if trigger_id == 'apply-calib-linear-button':
        if not baseline_points or len(baseline_points) < 2:
            return no_update, "错误: 线性拟合至少需要选择 2 个点", no_update
        return {'applied': True, 'type': 'linear', 'indices': sorted(baseline_points), 'method': method}, \
            f"状态: 已应用 [线性拟合] 校准\n算法: {method_name}\n拟合点数: {len(baseline_points)}", no_update

    return no_update, no_update, no_update


@app.callback(
    [Output('marked-points-store', 'data'),
     Output('previous-marked-points-store', 'data'),
     Output('last-action-store', 'data', allow_duplicate=True),
     Output('baseline-points-store', 'data')],
    Input('main-plot', 'clickData'),
    [State('marking-mode-store', 'data'), State('marked-points-store', 'data'), State('column-selector', 'value'),
     State('analysis-mode-store', 'data'), State('sampling-rate-input', 'value'),
     State('baseline-points-store', 'data')],
    prevent_initial_call=True
)
def handle_graph_click(click_data, marking_mode, current_points, selected_column, analysis_mode, sampling_rate,
                       baseline_points):
    if not click_data or marking_mode == 'none': return no_update, no_update, no_update, no_update

    previous_points = copy.deepcopy(current_points)
    time_val = click_data['points'][0]['x']
    sampling_rate = sampling_rate if sampling_rate is not None and sampling_rate > 0 else 1
    x_val = round(time_val * sampling_rate)

    if marking_mode == 'baseline':
        new_baseline_points = baseline_points if baseline_points else []
        if x_val not in new_baseline_points:
            new_baseline_points.append(x_val)
            new_baseline_points.sort()
        return no_update, no_update, no_update, new_baseline_points

    if analysis_mode == 'single':
        if not selected_column: return no_update, no_update, no_update, no_update
        y_val = click_data['points'][0]['y']
        if selected_column not in current_points['single'][marking_mode]: current_points['single'][marking_mode][
            selected_column] = []
        current_points['single'][marking_mode][selected_column].append((x_val, y_val))
    elif analysis_mode == 'multi':
        current_points['multi'][marking_mode].append(x_val)
        current_points['multi'][marking_mode] = sorted(list(set(current_points['multi'][marking_mode])))

    return current_points, previous_points, 'mark', no_update


@app.callback([Output('results-container', 'children'), Output('log-store', 'data'),
               Output('metric-annotations-store', 'data'), Output('results-data-store', 'data')],
              Input('calculate-button', 'n_clicks'),
              [State('marked-points-store', 'data'), State('column-selector', 'value'),
               State('analysis-mode-store', 'data'), State('sampling-rate-input', 'value')],
              prevent_initial_call=True)
def update_results(n_clicks, marked_points, selected_column, analysis_mode, sampling_rate):
    global processed_data
    if not n_clicks or processed_data is None: return no_update, no_update, no_update, no_update
    if analysis_mode == 'single' and not selected_column: return [
        html.P("请在单传感器模式下选择要分析的列。")], "", [], None
    results_data, log_data, annotations_data = calculate_metrics(marked_points, processed_data, selected_column,
                                                                 analysis_mode, sampling_rate)
    if isinstance(results_data, list) and results_data and isinstance(results_data[0],
                                                                      html.P): return results_data, log_data, annotations_data, None
    results_table = generate_results_table(results_data)
    header = html.Div([html.H4("指标计算结果"), html.Button("下载结果 (CSV)", id="download-results-button-real",
                                                            className="download-log-button")],
                      className="results-header")
    return [header, results_table], log_data, annotations_data, results_data


@app.callback(Output("download-results-csv", "data"), Input("download-results-button-real", "n_clicks"),
              State("results-data-store", "data"), prevent_initial_call=True)
def download_results_csv(n_clicks, results_data):
    if n_clicks and results_data:
        df = pd.DataFrame(results_data)
        return dcc.send_data_frame(df.to_csv, "calculation_results.csv", index=False)
    return no_update


@app.callback(
    [Output('marked-points-store', 'data', allow_duplicate=True),
     Output('results-container', 'children', allow_duplicate=True),
     Output('log-store', 'data', allow_duplicate=True),
     Output('metric-annotations-store', 'data', allow_duplicate=True),
     Output('previous-marked-points-store', 'data', allow_duplicate=True)],
    Input('clear-marks-button', 'n_clicks'), State('marked-points-store', 'data'), prevent_initial_call=True
)
def clear_all_markings(n_clicks, current_points):
    if not n_clicks: return no_update, no_update, no_update, no_update, no_update
    previous_points = copy.deepcopy(current_points)
    new_empty_points = {'single': {'peaks': {}, 'intake': {}, 'exhaust': {}},
                        'multi': {'peaks': [], 'intake': [], 'exhaust': []}}
    return new_empty_points, [], "", [], previous_points


@app.callback(
    [Output('marked-points-store', 'data', allow_duplicate=True),
     Output('last-action-store', 'data', allow_duplicate=True)],
    Input('undo-button', 'n_clicks'),
    [State('last-action-store', 'data'),
     State('previous-marked-points-store', 'data')],
    prevent_initial_call=True
)
def undo_last_action(n_clicks, last_action, prev_marked_points):
    if not n_clicks or not last_action:
        return no_update, no_update

    if last_action == 'mark' and prev_marked_points is not None:
        return prev_marked_points, None

    return no_update, no_update


# --- MODIFIED: 主更新回调函数，增加 1-R/R0 算法逻辑 ---
@app.callback(
    [Output("main-plot", "figure"), Output('column-selector', 'options'), Output('column-selector', 'value'),
     Output("upload-output", "children"), Output("trim-button", "disabled"), Output("download-button", "disabled"),
     Output("mark-peak-button", "disabled"), Output("mark-intake-button", "disabled"),
     Output("mark-exhaust-button", "disabled"), Output("end-marking-button", "disabled"),
     Output("undo-button", "disabled"), Output("calculate-button", "disabled"),
     Output("clear-marks-button", "disabled")],
    [Input("upload-data", "contents"), Input("trim-button", "n_clicks"), Input('column-selector', 'value'),
     Input('marked-points-store', 'data'), Input('analysis-mode-store', 'data'),
     Input('metric-annotations-store', 'data'), Input('sampling-rate-input', 'value'),
     Input('annotation-font-size-input', 'value'), Input('calibration-store', 'data'),
     Input('baseline-points-store', 'data')],
    [State("upload-data", "filename"), State("range-start", "value"), State("range-end", "value")],
)
def update_all(contents, trim_clicks, selected_col, marked_points, analysis_mode, annotations_data, sampling_rate,
               annotation_font_size, calib_params, baseline_points, filename, range_start, range_end):
    global original_data, processed_data
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "initial_load"
    col_options, col_value, upload_text = no_update, no_update, no_update

    if trigger_id == "initial_load":
        fig = create_figure(None, "Data Visualization and Marking", 'single')
        return (fig, [], None, "📂 Drag and Drop or Click to Upload CSV/Excel File", *[True] * 9)

    if trigger_id == "upload-data":
        if contents and filename:
            try:
                _, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                file_extension = os.path.splitext(filename)[-1].lower()
                if file_extension == '.csv':
                    original_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif file_extension in ['.xls', '.xlsx']:
                    original_data = pd.read_excel(io.BytesIO(decoded))
                else:
                    raise ValueError("Unsupported file type.")
                processed_data = original_data.copy()
                upload_text = f"✅ Uploaded: {filename}"
            except Exception as e:
                original_data, processed_data = None, None
                upload_text = f"❌ File parsing failed: {e}"
        else:
            original_data, processed_data = None, None
            upload_text = "📂 Drag and Drop or Click to Upload CSV/Excel File"

    if original_data is not None:
        start = int(range_start) if range_start is not None else 0
        end = int(range_end) if range_end is not None else len(original_data)
        temp_data = original_data.iloc[max(0, start):min(len(original_data), end)].reset_index(drop=True)

        if calib_params and calib_params.get('applied'):
            method = calib_params.get('method', 'div')
            calib_type = calib_params.get('type', 'constant')
            numeric_cols = temp_data.select_dtypes(include=np.number).columns

            if calib_type == 'constant':
                c_start, c_end = int(calib_params['range'][0]), int(calib_params['range'][1])
                if 0 <= c_start < c_end <= len(temp_data):
                    baseline_vals = temp_data.iloc[c_start:c_end][numeric_cols].mean()
                    if method == 'div':
                        baseline_vals = baseline_vals.replace(0, 1e-9)
                        temp_data[numeric_cols] = temp_data[numeric_cols] / baseline_vals
                    elif method == 'sub':
                        temp_data[numeric_cols] = temp_data[numeric_cols] - baseline_vals
                    elif method == 'one_minus_div':
                        baseline_vals = baseline_vals.replace(0, 1e-9)
                        temp_data[numeric_cols] = 1 - (temp_data[numeric_cols] / baseline_vals)

            elif calib_type == 'linear':
                indices = calib_params.get('indices', [])
                valid_indices = [i for i in indices if 0 <= i < len(temp_data)]
                if len(valid_indices) >= 2:
                    X_fit = np.array(valid_indices)
                    for col in numeric_cols:
                        Y_fit = temp_data[col].iloc[valid_indices].values
                        slope, intercept = np.polyfit(X_fit, Y_fit, 1)
                        baseline_curve = slope * temp_data.index + intercept

                        if method == 'div':
                            baseline_curve = np.where(np.abs(baseline_curve) < 1e-9, 1e-9, baseline_curve)
                            temp_data[col] = temp_data[col] / baseline_curve
                        elif method == 'sub':
                            temp_data[col] = temp_data[col] - baseline_curve
                        elif method == 'one_minus_div':
                            baseline_curve = np.where(np.abs(baseline_curve) < 1e-9, 1e-9, baseline_curve)
                            temp_data[col] = 1 - (temp_data[col] / baseline_curve)

        processed_data = temp_data

    if processed_data is not None:
        numeric_cols = processed_data.select_dtypes(include=np.number).columns.tolist()
        col_options = [{'label': col, 'value': col} for col in numeric_cols]
        if trigger_id == "upload-data":
            col_value = numeric_cols[0] if numeric_cols else None
            selected_col = col_value
        else:
            col_value = selected_col if selected_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    else:
        col_options, col_value = [], None

    has_data = processed_data is not None and not processed_data.empty
    current_selected_col = col_value if col_value is not no_update else selected_col

    main_fig = create_figure(processed_data, "Data Visualization and Marking", analysis_mode, current_selected_col,
                             marked_points, sampling_rate, annotations_data, annotation_font_size, baseline_points)
    disable_buttons = not has_data

    if trigger_id != "upload-data":
        upload_text = no_update

    return (main_fig, col_options, col_value, upload_text, *[disable_buttons] * 9)


@app.callback(Output("download-file", "data"), Input("download-button", "n_clicks"), State('upload-data', 'filename'),
              prevent_initial_call=True)
def download_file(n_clicks, filename):
    if processed_data is not None:
        base_name, _ = os.path.splitext(filename) if filename else ('processed_data', '')
        return dcc.send_data_frame(processed_data.to_csv, f"{base_name}_processed.csv", index=False)
    return None


# --- 运行应用的主入口 ---
if __name__ == "__main__":
    HOST, PORT = '127.0.0.1', 8040

    css_string = """
    html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; font-family: Segoe UI, sans-serif; background-color: #f8f9fa; }
    #app-container { display: flex; flex-direction: column; height: 100vh; max-width: 1600px; margin: auto; padding: 20px; box-sizing: border-box; }
    #header { flex-shrink: 0; text-align: center; margin-bottom: 10px; }
    #header h1 { color: #333; }
    #main-content { display: flex; gap: 20px; flex-wrap: nowrap; align-items: flex-start; flex-grow: 1; overflow-y: auto; min-height: 0; padding-bottom: 20px; }
    #control-panel { flex: 0 0 380px; display: flex; flex-direction: column; gap: 20px; position: sticky; top: 0; }
    #graph-container { flex-grow: 1; min-width: 500px; display: flex; flex-direction: column; gap: 20px; }
    .control-card, .graph-card, .results-card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 20px; }
    .control-card h3, .graph-card h3, .results-card h4 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; color: #343a40; }
    .results-card { max-height: 400px; overflow-y: auto; }
    .control-group { margin-bottom: 15px; }
    .control-group label { display: block; font-weight: bold; margin-bottom: 5px; font-size: 0.9em; }
    .control-group input[type=number], .control-group input[type=text] { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    .custom-tabs { border-bottom: 1px solid #dee2e6; }
    .custom-tab { padding: 12px; border-top: 3px solid transparent; color: #495057; cursor: pointer; }
    .custom-tab--selected { border-top: 3px solid #007bff; color: #007bff; font-weight: bold; background-color: #fff; border-bottom: 1px solid transparent; }
    #upload-data { border: 2px dashed #007bff; border-radius: 5px; padding: 20px; text-align: center; cursor: pointer; transition: background-color 0.2s; }
    #upload-data:hover { background-color: #e9f5ff; }
    #upload-output { color: #007bff; font-weight: bold; }
    button:disabled { background-color: #ccc !important; color: #666 !important; cursor: not-allowed; }
    #control-panel button { color: white; background-color: #0069d9; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; transition: background-color 0.2s; width: 100%; box-sizing: border-box; font-weight: bold; }
    #control-panel button:hover:not(:disabled) { background-color: #5a6268; }
    .marking-controls { display: flex; gap: 8px; flex-wrap: nowrap; margin-top: 15px; overflow-x: auto; padding-bottom: 5px; }
    .marking-controls button { color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; transition: background-color 0.2s; font-weight: bold; flex-shrink: 0; }
    .btn-peak { background-color: #dc3545; } .btn-peak:hover:not(:disabled) { background-color: #c82333; }
    .btn-intake { background-color: #28a745; } .btn-intake:hover:not(:disabled) { background-color: #218838; }
    .btn-exhaust { background-color: #17a2b8; } .btn-exhaust:hover:not(:disabled) { background-color: #138496; }
    .btn-end { background-color: #6c757d; } .btn-end:hover:not(:disabled) { background-color: #5a6268; }
    .btn-undo { background-color: #ffc107; color: #212529; } .btn-undo:hover:not(:disabled) { background-color: #e0a800; }
    .btn-calculate { background-color: #007bff; } .btn-calculate:hover:not(:disabled) { background-color: #0069d9; }
    .btn-clear { background-color: #343a40; } .btn-clear:hover:not(:disabled) { background-color: #23272b; }
    .results-header button { color: white; background-color: #28a745; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; transition: background-color 0.2s; font-weight: bold; width: auto; }
    .results-header button:hover:not(:disabled) { background-color: #218838; }
    .column-selector-wrapper { margin-bottom: 15px; }
    .graph-card { display: flex; flex-direction: column; justify-content: center; }
    #main-plot { width: 100%; max-width: 100%; aspect-ratio: 16 / 9; max-height: 70vh; }
    .results-header { display: flex; justify-content: space-between; align-items: center; }
    .results-table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
    .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    .results-table th { background-color: #f2f2f2; font-weight: bold; }
    .results-table tr:nth-child(even) { background-color: #f9f9f9; }
    .results-table tr:hover { background-color: #f1f1f1; }
    """

    if not os.path.exists(assets_path):
        os.makedirs(assets_path)

    with open(os.path.join(assets_path, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_string)

    threading.Timer(1, lambda: webbrowser.open_new(f"http://{HOST}:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False)
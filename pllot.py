import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import signal


df = pd.read_csv("output_for_plot.csv")
dfa = df[df["line"] == "A, 31"].sort_values(by=["time"], ascending=[True])
dfa = pd.concat([dfa[:100], dfa[:120]])
pre = dfa["0.5"].to_numpy()
gt = dfa["gt"].to_numpy()
upper_ci = dfa["0.98"].to_numpy()
lower_ci = dfa["0.02"].to_numpy()
x = np.arange(len(pre))
base_value = 14.5
baseline = np.full((len(pre)), base_value)
start_x = int(len(x) / 4 * 3)
frame_duration = 50
# upper_ci = signal.savgol_filter(upper_ci, 30, 3)


def roll_plot(speed=1):
    frame_num = int((len(x) - start_x) / speed)
    print(f"len x={len(x)}, start-x={start_x}, frame-num={frame_num}")
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=gt, line=dict(width=3), name="in(log)"),
            go.Scatter(x=x, y=pre, line_color="rgba(200, 100, 0, 1)", name="predict"),
            go.Scatter(x=x, y=upper_ci, line_color="rgba(0,0,0,0)", showlegend=False),
            go.Scatter(x=x, y=lower_ci, line_color="rgba(0,0,0,0)", fill='tonexty', fillcolor='rgba(200, 100, 0, 0.1)',
                       name="confidence interval"),
            go.Scatter(x=x, y=baseline, mode="lines", line_color="rgba(50, 200, 50, 1)", name="baseline")
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, start_x], autorange=False),
            yaxis=dict(range=[0, 20], autorange=True),
            title="A, 31",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": False
                                    },
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ]
                        )
                    ]
                )
            ]
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=x-speed*k, y=gt, line=dict(width=3), name="in(log)"),
                    go.Scatter(x=x-speed*k, y=pre, line_color="rgba(200, 100, 0, 1)", name="predict"),
                    go.Scatter(x=x-speed*k, y=upper_ci, line_color="rgba(0,0,0,0)", showlegend=False),
                    go.Scatter(x=x-speed*k, y=lower_ci, line_color="rgba(0,0,0,0)", fill='tonexty',
                               fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval")
                ]
                # 'tonexty' == To Next Y. ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
            )
            for k in range(frame_num)
        ]
    )

    fig.show()


def fade_in_plot(speed=1):
    end_x = int(len(x) / 4)
    frame_num = int((len(x) - end_x) / speed)
    fig = make_subplots(1, 1)

    # Add Initial lines
    fig.add_trace(go.Scatter(x=x[:end_x], y=gt[:end_x], mode="lines", line=dict(width=3, smoothing=1), name="truth"))
    fig.add_trace(go.Scatter(x=x[end_x-1:end_x], y=pre[end_x-1:end_x], mode="lines", line=dict(color="rgba(200, 100, 0, 1)"), name="predict"))
    fig.add_trace(go.Scatter(x=x[end_x-1:end_x], y=upper_ci[end_x-1:end_x], line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=x[end_x-1:end_x], y=lower_ci[end_x-1:end_x], line=dict(color="rgba(0,0,0,0)"), fill='tonexty', fillcolor='rgba(200, 100, 0, 0.1)',
                       name="confidence interval"))
    fig.add_trace(go.Scatter(y=baseline, mode="lines", line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold"))
    fig.add_trace(go.Scatter(y=baseline, mode="lines", line=dict(color="rgba(40, 40, 40, 1)", dash="dot", shape="hv"), name="baseline"))

    # Get threshold
    upper_ci_smooth = signal.savgol_filter(upper_ci[end_x::speed], 30, 2)
    thres_mat = np.tile(upper_ci_smooth, (len(x), 1)).T
    thres_mat = np.clip(thres_mat, base_value, 15.7)

    # Add Frames
    fig.frames = [
            go.Frame(
                data=[
                    go.Scatter(x=x[:end_x + speed * k], y=gt[:end_x + speed * k], line=dict(width=3), name="in(log)"),
                    go.Scatter(x=x[end_x:end_x + speed * k], y=pre[end_x:end_x + speed * k], line_color="rgba(200, 100, 0, 1)", name="predict"),
                    go.Scatter(x=x[end_x:end_x + speed * k], y=upper_ci[end_x:end_x + speed * k], line_color="rgba(0,0,0,0)", showlegend=False),
                    go.Scatter(x=x[end_x:end_x + speed * k], y=lower_ci[end_x:end_x + speed * k], line_color="rgba(0,0,0,0)", fill='tonexty',
                               fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval"),
                    go.Scatter(y=thres_mat[k], mode="lines",
                               line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold")
                ]
                # 'tonexty' == To Next Y. ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
            )
            for k in range(frame_num)
        ]
    # Add layout
    fig.update_layout(
        xaxis=dict(range=[0, len(x)], autorange=False),
        yaxis=dict(range=[13.5, 16.5], autorange=False),
        title="Web Traffic Forecast",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": frame_duration,
                                    "redraw": False
                                },
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    )
                ]
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    fade_in_plot(1)
    # roll_plot(1)

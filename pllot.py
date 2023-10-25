import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
import numpy as np
from scipy import signal
import cv2


df = pd.read_csv("output_for_plot.csv")
dfa = df[df["line"] == "A, 31"].sort_values(by=["time"], ascending=[True])
# dfa = pd.concat([dfa[:100], dfa[:120]])
dfa = dfa[250:450]
pre = dfa["0.5"].to_numpy()
gt = dfa["gt"].to_numpy()
upper_ci = dfa["0.98"].to_numpy()
lower_ci = dfa["0.02"].to_numpy()
yrange = [min(min(pre), min(gt), min(lower_ci)) - 0.5, max(max(pre), max(gt), max(upper_ci)) + 0.5]
x = np.arange(len(pre))
base_value = 14.75
start_x = int(len(x) / 4 * 3)
frame_duration = 50
gt_delay = 10  # num of frames
# interpolate
x_interp = np.arange(0, len(pre), 0.5)
pre = np.interp(x_interp, x, pre)
gt = np.interp(x_interp, x, gt)
upper_ci = np.interp(x_interp, x, upper_ci)
lower_ci = np.interp(x_interp, x, lower_ci)
x = np.arange(len(x_interp))

baseline = np.full((len(x)), base_value)
xtickval = [0, 96, 192, 288, 384]
xticktext = ["2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-23"]

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
    fig.add_trace(go.Scatter(x=x[end_x:end_x+1], y=pre[end_x:end_x+1], mode="lines", line=dict(color="rgba(200, 100, 0, 1)"), name="predict"))
    fig.add_trace(go.Scatter(x=x[end_x:end_x+1], y=upper_ci[end_x:end_x+1], line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=x[end_x:end_x+1], y=lower_ci[end_x:end_x+1], line=dict(color="rgba(0,0,0,0)"), fill='tonexty', fillcolor='rgba(200, 100, 0, 0.1)',
                       name="confidence interval"))
    fig.add_trace(go.Scatter(y=baseline, mode="lines", line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold"))
    fig.add_trace(go.Scatter(y=baseline, mode="lines", line=dict(color="rgba(40, 40, 40, 1)", dash="dot", shape="hv"), name="baseline"))

    # Get threshold
    upper_ci_smooth = signal.savgol_filter(upper_ci[end_x::speed], 30, 3)
    thres_mat = np.tile(upper_ci_smooth, (len(x), 1)).T
    thres_mat = np.clip(thres_mat, base_value, None)

    # Add Frames
    frames = []
    for f_first in range(gt_delay):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x[:end_x], y=gt[:end_x],
                           line=dict(width=3), name="truth"),
                go.Scatter(x=x[end_x:end_x + speed * f_first], y=pre[end_x:end_x + speed * f_first],
                           line_color="rgba(200, 100, 0, 1)", name="predict"),
                go.Scatter(x=x[end_x:end_x + speed * f_first], y=upper_ci[end_x:end_x + speed * f_first],
                           line_color="rgba(0,0,0,0)", showlegend=False, name="CI upper"),
                go.Scatter(x=x[end_x:end_x + speed * f_first], y=lower_ci[end_x:end_x + speed * f_first],
                           line_color="rgba(0,0,0,0)", fill='tonexty',
                           fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval"),
                go.Scatter(y=thres_mat[f_first], mode="lines",
                           line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold")
            ]
        ))
    for f_mid in range(gt_delay, frame_num):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x[:end_x + speed * (f_mid-gt_delay)], y=gt[:end_x + speed * (f_mid-gt_delay)], line=dict(width=3), name="truth"),
                go.Scatter(x=x[end_x:end_x + speed * f_mid], y=pre[end_x:end_x + speed * f_mid],
                           line_color="rgba(200, 100, 0, 1)", name="predict"),
                go.Scatter(x=x[end_x:end_x + speed * f_mid], y=upper_ci[end_x:end_x + speed * f_mid],
                           line_color="rgba(0,0,0,0)", showlegend=False, name="CI upper"),
                go.Scatter(x=x[end_x:end_x + speed * f_mid], y=lower_ci[end_x:end_x + speed * f_mid],
                           line_color="rgba(0,0,0,0)", fill='tonexty',
                           fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval"),
                go.Scatter(y=thres_mat[f_mid], mode="lines",
                           line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold")
            ]
        ))
    for f_last in range(frame_num, frame_num+gt_delay):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x[:end_x + speed * (f_last-gt_delay)], y=gt[:end_x + speed * (f_last-gt_delay)], line=dict(width=3), name="truth"),
                go.Scatter(x=x[end_x:], y=pre[end_x:],
                           line_color="rgba(200, 100, 0, 1)", name="predict"),
                go.Scatter(x=x[end_x:], y=upper_ci[end_x:],
                           line_color="rgba(0,0,0,0)", showlegend=False, name="CI upper"),
                go.Scatter(x=x[end_x:], y=lower_ci[end_x:],
                           line_color="rgba(0,0,0,0)", fill='tonexty',
                           fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval"),
                go.Scatter(y=thres_mat[-1], mode="lines",
                           line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold")
            ]
        ))
    # fig.frames = [
    #         go.Frame(
    #             data=[
    #                 go.Scatter(x=x[:end_x + speed * k], y=gt[:end_x + speed * k], line=dict(width=3), name="truth"),
    #                 go.Scatter(x=x[end_x:end_x + speed * k], y=pre[end_x:end_x + speed * k], line_color="rgba(200, 100, 0, 1)", name="predict"),
    #                 go.Scatter(x=x[end_x:end_x + speed * k], y=upper_ci[end_x:end_x + speed * k], line_color="rgba(0,0,0,0)", showlegend=False, name="CI upper"),
    #                 go.Scatter(x=x[end_x:end_x + speed * k], y=lower_ci[end_x:end_x + speed * k], line_color="rgba(0,0,0,0)", fill='tonexty',
    #                            fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval"),
    #                 go.Scatter(y=thres_mat[k], mode="lines",
    #                            line=dict(color="rgba(100, 200, 100, 1)", dash="dash", shape="hv"), name="threshold")
    #             ]
    #             # 'tonexty' == To Next Y. ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
    #         )
    #         for k in range(frame_num)
    #     ]
    fig.frames = frames
    # Add layout
    fig.update_layout(
        xaxis=dict(range=[0, len(x)], autorange=False, tickmode="array", tickvals=xtickval, ticktext=xticktext),
        yaxis=dict(range=yrange, autorange=False, tickvals=[]),
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

    # for i, frame in enumerate(frames):
    #     fig.update(frames=[go.Frame(data=frame.data)])
    #     fig_path = f'plot_frames/frame-{i}.png'
    #     plotly.io.write_image(fig, fig_path, format='png', width=1024, height=768)

    fig.write_html("predict-sample.html", auto_play=False)
    # fig.show()


if __name__ == "__main__":
    fade_in_plot(1)
    # roll_plot(1)

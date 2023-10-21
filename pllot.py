import pandas as pd
import plotly.graph_objects as go
import numpy as np


df = pd.read_csv("output_for_plot.csv")
dfa = df[df["line"] == "B, 33"].sort_values(by=["time"], ascending=[True])
pre = dfa["0.5"]
gt = dfa["gt"]
upper_ci = dfa["0.98"]
lower_ci = dfa["0.02"]
x = np.arange(len(pre))
start_x = int(len(x) / 4 * 3)
frame_duration = 20


def roll_plot(speed=1):
    frame_num = int((len(x) - start_x) / speed)
    print(f"len x={len(x)}, start-x={start_x}, frame-num={frame_num}")
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=gt, line=dict(width=3), name="in(log)"),
            go.Scatter(x=x, y=pre, line_color="rgba(200, 100, 0, 1)", name="predict"),
            go.Scatter(x=x, y=upper_ci, line_color="rgba(0,0,0,0)", showlegend=False),
            go.Scatter(x=x, y=lower_ci, line_color="rgba(0,0,0,0)", fill='tonexty', fillcolor='rgba(200, 100, 0, 0.1)',
                       name="confidence interval")
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
    end_x = int(len(x) / 10)
    frame_num = int((len(x) - end_x) / speed)
    fig = go.Figure(
        data=[
            go.Scatter(x=x[:end_x], y=gt[:end_x], line=dict(width=3), name="in(log)"),
            go.Scatter(x=x[:end_x], y=pre[:end_x], line_color="rgba(200, 100, 0, 1)", name="predict"),
            go.Scatter(x=x[:end_x], y=upper_ci[:end_x], line_color="rgba(0,0,0,0)", showlegend=False),
            go.Scatter(x=x[:end_x], y=lower_ci[:end_x], line_color="rgba(0,0,0,0)", fill='tonexty', fillcolor='rgba(200, 100, 0, 0.1)',
                       name="confidence interval")
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, len(x)], autorange=False),
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
                    go.Scatter(x=x[:end_x + speed * k], y=gt[:end_x + speed * k], line=dict(width=3), name="in(log)"),
                    go.Scatter(x=x[:end_x + speed * k], y=pre[:end_x + speed * k], line_color="rgba(200, 100, 0, 1)", name="predict"),
                    go.Scatter(x=x[:end_x + speed * k], y=upper_ci[:end_x + speed * k], line_color="rgba(0,0,0,0)", showlegend=False),
                    go.Scatter(x=x[:end_x + speed * k], y=lower_ci[:end_x + speed * k], line_color="rgba(0,0,0,0)", fill='tonexty',
                               fillcolor='rgba(200, 100, 0, 0.1)', name="confidence interval")
                ]
                # 'tonexty' == To Next Y. ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
            )
            for k in range(frame_num)
        ]
    )

    fig.show()


if __name__ == "__main__":
    # fade_in_plot(1)
    roll_plot(1)

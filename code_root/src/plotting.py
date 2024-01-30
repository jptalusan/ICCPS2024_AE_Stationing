import re
import dateparser
import os
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px

RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
PLOTS_DIR = "./plots"
stops_path = f"{PLOTS_DIR}/all_stops.csv"
buses_path = f"{PLOTS_DIR}/all_buses.csv"

def parse_filename(df):
    df[['date', 'method', 'depot', 'real', 'test', 'noise', 'iter']] = df['filename'].str.split('_', expand=True)
    return df

def plot_figure_4A():
    sdf = pd.read_csv(stops_path, index_col=0)
    bdf = pd.read_csv(buses_path, index_col=0)
    # parse filename
    sdf = parse_filename(sdf)
    bdf = parse_filename(bdf)

    bdf = bdf.groupby("filename").agg({"total_served":"sum", "service_kms":"sum", "total_deadsecs":"sum"})
    bdf = parse_filename(bdf.reset_index())

    # drop nosub for this plot
    bdf = bdf[bdf['method'] != "nosub"]
    bdf = bdf[bdf['iter'] != "50"]

    bdf = bdf.groupby("depot").agg({"service_kms":list}).reset_index()
    bdf["std"] = bdf["service_kms"].apply(lambda x: np.std(x))
    bdf["mean"] = bdf["service_kms"].apply(lambda x: np.mean(x))
    bdf["count"] = bdf["service_kms"].apply(len)
    bdf["std_e"] = bdf["std"] / bdf["count"].apply(np.sqrt)

    sort_dict = {"garage": 0, "agency": 1, "search": 2, "mcts": 3}
    bdf = bdf.iloc[bdf["depot"].map(sort_dict).sort_values().index]
    bdf.depot = bdf['depot'].str.upper()
    bdf["depot"] = pd.Categorical(bdf["depot"], ["GARAGE", "AGENCY", "SEARCH", "MCTS"])
    bdf = bdf.sort_values("depot")

    fig = px.line(bdf, x="depot", y="mean", error_y="std_e", markers=True, title="Line Plot with Error Bars")
    fig.update_traces(line={"width": 5})
    fig.data[0].error_y.thickness = 5
    fig.update_xaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_layout(
        width=800,
        height=500,
        # plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
        # paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent paper background
        plot_bgcolor="white",  # Transparent background
        paper_bgcolor="white",  # Transparent paper background
        margin=dict(l=5, r=5, t=5, b=5),
        title="",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            color="black",
        ),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)  # Change x and y-axis labels
    fig.update_xaxes(title_text="Stationing Plan")
    fig.update_yaxes(title_text="Deadhead kilometers")
    fig.write_image(f"{PLOTS_DIR}/figure_4_bottom.png", scale=3)
    return True

def plot_figure_4B():
    sdf = pd.read_csv(stops_path, index_col=0)
    bdf = pd.read_csv(buses_path, index_col=0)
    # parse filename
    sdf = parse_filename(sdf)
    bdf = parse_filename(bdf)
    df = sdf.groupby("filename").agg({"total_passenger_ons":"sum", "total_passenger_offs":"sum", "total_passenger_walk_away":"sum"})
    df = parse_filename(df.reset_index())

    # drop nosub for this plot
    df = df[df['method'] != "nosub"]
    df = df[df['iter'] != "50"]

    df = df.groupby("depot").agg({"total_passenger_ons":list}).reset_index()
    df["std"] = df["total_passenger_ons"].apply(lambda x: np.std(x))
    df["mean"] = df["total_passenger_ons"].apply(lambda x: np.mean(x))
    df["count"] = df["total_passenger_ons"].apply(len)
    df["std_e"] = df["std"] / df["count"].apply(np.sqrt)
    df["mean_log"] = np.log10(df["mean"])  # Logarithm with base 10
    sort_dict = {"garage": 0, "agency": 1, "search": 2, "mcts": 3}
    df = df.iloc[df["depot"].map(sort_dict).sort_values().index]
    df.depot = df['depot'].str.upper()
    df["depot"] = pd.Categorical(df["depot"], ["GARAGE", "AGENCY", "SEARCH", "MCTS"])
    df = df.sort_values("depot")
    fig = px.line(
        df,
        x="depot",
        y="mean",
        # error_y="std_e",
        markers=True,
        title="Line Plot with Error Bars"
        # , facet_col="type"
    )
    fig.update_traces(line={"width": 5})
    fig.data[0].error_y.thickness = 5
    # fig.data[1].error_y.thickness = 5
    fig.update_xaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_layout(
        width=600,
        height=500,
        # plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
        # paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent paper background
        plot_bgcolor="white",  # Transparent background
        paper_bgcolor="white",  # Transparent paper background
        margin=dict(l=5, r=5, t=5, b=5),
        title="",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            color="black",
        ),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)  # Change x and y-axis labels
    # fig.update_xaxes(title_text="Stationing and Dispatch Plan")
    fig.update_yaxes(title_text="Mean passengers served")


    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    # turn off axis titels of all axes
    fig.for_each_xaxis(lambda x: x.update({"title": ""}))
    # fig.for_each_yaxis(lambda y: y.update({'title': ''}))
    fig.add_annotation(
        showarrow=False, xanchor="center", xref="paper", x=0.5, yref="paper", y=-0.15, text="Stationing Plans"
    )
    fig.write_image(f"{PLOTS_DIR}/figure_4_top.png", scale=3)
    return True

def plot_figure_5():
    sdf = pd.read_csv(stops_path, index_col=0)
    sdf = parse_filename(sdf)
    sdf = sdf[sdf['iter'] != "50"]
    sdf["method"] = pd.Categorical(sdf["method"], ["mcts", "baseline", "nosub"])
    sdf = sdf.sort_values("method")
        
    methods = []
    heatmap_arr = []
    for k, v in sdf.groupby("filename"):
        if v['date'].iloc[0] != "20210312":
            continue
        v = v.sort_values(by="total_passenger_ons", ascending=False).head(25)
        names = v.stop_id.tolist()
        names = names[1:]
        top_25 = v.total_passenger_ons.to_numpy()
        top_25 = top_25[1:]
        heatmap_arr.append(top_25)
        if v['method'].iloc[0] == "mcts":
            method = "mcts"
        elif v['method'].iloc[0] == "nosub":
            method = "no subsitute"
        else:
            method = v['method'].iloc[0] + ":" + v['depot'].iloc[0]
        methods.append(method)
        
    fig = px.imshow(heatmap_arr,
                labels=dict(x="Stops", y="Method", color="Ridership"),
                x=names,
                y=methods
               )
    fig.update_xaxes(side="top")
    fig.update_xaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_layout(
        width=1000,
        height=400,
        # plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
        # paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent paper background
        plot_bgcolor="white",  # Transparent background
        paper_bgcolor="white",  # Transparent paper background
        margin=dict(l=5, r=5, t=5, b=5),
        title="",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            color="black",
        ),
        coloraxis_colorbar=dict(len=0.5, thickness=20)
    )

    fig.write_image(f"{PLOTS_DIR}/figure_5.png", scale=3)

    return True

def read_last_n_lines(file_path, n):
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_n_lines = lines[-n:]
    return last_n_lines

def parse_stream_log(log_entry):
    # Define a regex pattern for timestamp and message
    pattern = re.compile(r'\[(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\[D\] (.*)')

    # Find all matches in the log entry
    match = pattern.match(log_entry)

    # Extract timestamp and message
    if match:
        timestamp = match.group(1)
        timestamp = dateparser.parse(timestamp)
        message = match.group(2)
        # print(f"Timestamp: {timestamp}, Message: {message}")
        return timestamp, message
    else:
        # print("No match found.")
        return None, None
    
    
def plot_figure_7():
    # String to match in filenames
    string_to_match = "mcts_mcts"

    # Loop through all subdirectories
    data_arr = []

    for subdir, _, files in os.walk(LOGS_DIR):
        if string_to_match not in subdir:
            continue
        for file in files:
            if file == "stream.log":
                file_path = os.path.join(subdir, file)
                filename = subdir.split("/")[-1]
                filename = filename.split("_")
                # print(filename)
                date = filename[0]
                method = filename[1]
                iter = filename[-1]
                try:
                    # Open and read the contents of the file
                    with open(file_path, 'r') as results_file:
                        contents = results_file.readlines()
                        first_line = contents[0]
                        last_line = contents[-1]
                        decision_epochs = contents[-3]
                        
                        start_time, _ = parse_stream_log(first_line)
                        end_time, _ = parse_stream_log(last_line)
                        _, decision_counts = parse_stream_log(decision_epochs)
                        
                        decision_counts = decision_counts.split(":")[-1]
                        decision_counts = int(decision_counts)
                        duration = (end_time - start_time).total_seconds()
                        # print(duration, decision_counts)
                        
                        res = {"date":date, "method":method, "iter":iter, "duration": duration, "decisions":decision_counts}
                        data_arr.append(res)
                        # print(f"Contents of {file_path}:\n{first_line}\n{last_line}\n{decision_epochs}")
                except Exception as e:
                    # print(f"Error reading {file_path}: {e}")
                    pass
                
    if len(data_arr) == 0:
        return False
                
    df = pd.DataFrame(data_arr)
    df["time_per_decision"] = df["duration"] / df["decisions"]
    df = df.groupby("iter").agg({"duration":list, "time_per_decision":list}).reset_index()

    df["duration_std"] = df["duration"].apply(lambda x: np.std(x))
    df["duration_mean"] = df["duration"].apply(lambda x: np.mean(x))
    df["count"] = df["duration"].apply(len)
    df["duration_std_e"] = df["duration_std"] / df["count"].apply(np.sqrt)


    df["time_per_decision_std"] = df["time_per_decision"].apply(lambda x: np.std(x))
    df["time_per_decision_mean"] = df["time_per_decision"].apply(lambda x: np.mean(x))
    df["count"] = df["duration"].apply(len)
    df["time_per_decision_std_e"] = df["time_per_decision_std"] / df["count"].apply(np.sqrt)


    df.iter = df.iter.astype('int')
    df = df.sort_values(by="iter")

    # Create subplot figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Time per decision', 'Total run time'],
                        vertical_spacing=0.05)

    # Add traces to the subplots using plotly express
    trace_per_decision = px.line(
        df,
        x="iter",
        y="time_per_decision_std",
        error_y="time_per_decision_std_e",
        markers=True,
        title="Line Plot with Error Bars"
        # , facet_col="type"
    )

    trace_duration = px.line(
        df,
        x="iter",
        y="duration_mean",
        error_y="duration_std_e",
        markers=True,
        title="Line Plot with Error Bars"
        # , facet_col="type"
    )
    trace1 = trace_per_decision.update_traces(showlegend=False).data[0]
    trace2 = trace_duration.update_traces(showlegend=False).data[0]

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)

    fig.update_traces(line={"width": 5})

    fig.data[0].error_y.thickness = 5
    fig.data[1].error_y.thickness = 5

    fig.update_xaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor="lightgrey", title="Seconds")

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)  # Change x and y-axis labels

    # Update layout
    fig.update_layout(
        width=600,
        height=600,
        # plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
        # paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent paper background
        plot_bgcolor="white",  # Transparent background
        paper_bgcolor="white",  # Transparent paper background
        margin=dict(l=5, r=5, t=20, b=5),
        # title="",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            color="black",
        ),
        xaxis_title='', xaxis2_title='Iterations'
    )

    # Show the plot
    fig.write_image(f"{PLOTS_DIR}/figure_7.png", scale=3)

    return True

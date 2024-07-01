import numpy as np
import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    LogColorMapper,
    WheelZoomTool,
    Grid,
    FixedTicker,
    FuncTickFormatter,
    Arrow,
    OpenHead,
    CustomJS,
    BasicTickFormatter,
    NumeralTickFormatter,
    RangeTool,
)
from bokeh.palettes import RdYlBu11
from bokeh.plotting import figure, column
from bokeh.transform import dodge
from bokeh.events import RangesUpdate, Tap
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure, show
from bokeh.palettes import Accent8
from math import pi

from ._util import (
    get_size_hover_formatter,
    get_size_tick_formatter,
    get_time_tick_formatter,
    get_time_hover_formatter,
    get_percent_tick_formatter,
    get_percent_hover_formatter,
    get_trimmed_tick_formatter,
    plot,
    get_html_tooltips,
    get_height,
    get_factor_cmap,
    get_palette,
    # clamp,
    trimmed,
)

import datashader as ds
from ..graphframe import GraphFrame

def timeline(trace_events, start_time, end_time, show_depth=False, instant_events=False):
    """Displays the events of a trace against time

    Instant events are represented by points, functions are represented by horizontal
    bars, and MPI messages are represented by lines connecting send/receive events."""

    # Generate necessary metrics
    # trace.calc_exc_metrics(["Timestamp (ns)"])
    # trace._match_events()
    # trace._match_caller_callee()

    min_ts = start_time
    max_ts = end_time

    # Prepare data for plotting
    # events = (
    #     trace_events[
    #         trace_events["Event Type"].isin(
    #             ["Enter", "Instant"] if instant_events else ["Enter"]
    #         )
    #     ]
    #     .sort_values(by="time.inc", ascending=False)
    #     .copy(deep=False)
    # )
    events = trace_events
    events["_depth"] = events["_depth"].astype(float).fillna("")

    # Determine y-coordinates from process and depth
    y_tuples = (
        list(zip(events["column_name"]))
    )

    codes, uniques = pd.factorize(y_tuples, sort=True)
    events["y"] = codes
    num_ys = len(uniques)

    depth_ticks = np.arange(0, num_ys)
    process_ticks = np.array(
        [i for i, v in enumerate(uniques) if len(v) == 1 or v[1] == 0]
    )

    # Define CDS for glyphs to be empty
    hbar_source = ColumnDataSource(events.head(1))
    scatter_source = ColumnDataSource(events.head(0))
    image_source = ColumnDataSource(
        data=dict(
            image=[np.zeros((50, 16), dtype=np.uint32)], x=[0], y=[0], dw=[0], dh=[0]
        )
    )

   
    # Callback function that updates CDS
    def update_cds(event):
        x0 = event.x0 if event is not None else min_ts
        x1 = event.x1 if event is not None else max_ts

        x0 = x0 - (x1 - x0) * 0.25
        x1 = x1 + (x1 - x0) * 0.25

        # Remove events that are out of bounds
        in_bounds = events[
            (
                (events["start_time"] < x1) & (events["end_time"] > x0)
            )
        ].copy(deep=False)

        # Update hbar_source to keep 5000 largest functions
        func = in_bounds
        large = func
        # large = func.head(5000)
        hbar_source.data = large

        
        # Rasterize the rest
        small = func.tail(len(func) - 5000).copy(deep=True)
        small["group_name"] = small["group_name"].astype("str")
        small["group_name"] = small["group_name"].astype("category")

        # if len(small):
        #     # Create a new Datashader canvas based on plot properties
        #     cvs = ds.Canvas(
        #         plot_width=1920 if p.inner_width == 0 else p.inner_width,
        #         plot_height=num_ys,
        #         x_range=(x0, x1),
        #         y_range=(-0.5, num_ys - 0.5),
        #     )

        #     # Feed the data into datashader
        #     agg = cvs.points(small, x="start_time", y="y", agg=ds.count_cat("group_name"))

        #     # Generate image
        #     img = ds.tf.shade(
        #         agg,
        #         color_key=get_palette(trace_events, 0.7),
        #     )

        #     # Update CDS
        #     image_source.data = dict(
        #         image=[np.flipud(img.to_numpy())],
        #         x=[x0],
        #         y=[num_ys - 0.5],
        #         dw=[x1 - x0],
        #         dh=[num_ys],
        #     )
        # else:
        #     image_source.data = dict(
        #         image=[np.zeros((50, 16), dtype=np.uint32)],
        #         x=[x0],
        #         y=[num_ys - 0.5],
        #         dw=[x1 - x0],
        #         dh=[num_ys],
        #     )
        image_source.data = dict(
            image=[np.zeros((50, 16), dtype=np.uint32)],
            x=[x0],
            y=[num_ys - 0.5],
            dw=[x1 - x0],
            dh=[num_ys],
        )

    # Create Bokeh plot
    # min_height = 50 + 22 * len(events["Name"].unique())
    plot_height = 60 + 22 * num_ys
    # height = clamp(plot_height, min_height, 900)

    p = figure(
        x_range=(min_ts, max_ts + (max_ts - min_ts) * 0.05),
        y_range=(num_ys - 0.5, -0.5),
        x_axis_location="above",
        tools="hover,xpan,reset,xbox_zoom,xwheel_zoom,save",
        output_backend="webgl",
        height=min(500, plot_height),
        sizing_mode="stretch_width",
        toolbar_location=None,
        # x_axis_label="Time",
    )

    # p.min_border_bottom = height - plot_height

    # Create color maps
    fill_cmap = get_factor_cmap("group_name", trace_events)
    line_cmap = get_factor_cmap("group_name", trace_events, scale=0.7)

    # Add lines for each process
    # p.segment(
    #     x0=[0] * len(process_ticks),
    #     x1=[max_ts] * len(process_ticks),
    #     y0=process_ticks,
    #     y1=process_ticks,
    #     line_dash="dotted",
    #     line_color="gray",
    # )

    # Add bars for large functions
    hbar = p.hbar(
        left="start_time",
        right="end_time",
        y="y",
        height=0.8 if show_depth else 0.8,
        source=hbar_source,
        fill_color=fill_cmap,
        line_color=line_cmap,
        line_width=1,
        line_alpha=0.5,
        legend_field="group_name",
    )

    # Add raster for small functions
    p.image_rgba(source=image_source)

    # Add custom grid lines
    # p.xgrid.visible = False
    p.ygrid.visible = False

    g1 = Grid(
        dimension=1,
        grid_line_color="white",
        grid_line_width=2 if show_depth else 2,
        ticker=FixedTicker(
            ticks=np.concatenate([depth_ticks - 0.49, depth_ticks + 0.49])
        ),
        level="glyph",
    )
    g2 = Grid(
        dimension=1,
        grid_line_width=2,
        # band_fill_color="gray",
        # band_fill_alpha=0.1,
        ticker=FixedTicker(ticks=process_ticks - 0.5),
        level="glyph",
    )
    p.add_layout(g1)
    p.add_layout(g2)

    # Additional plot config
    p.xaxis.formatter = get_time_tick_formatter()
    p.yaxis.formatter = FuncTickFormatter(
        args={
            "uniques": uniques,
        },
        code="""
            return uniques[Math.floor(tick)][0];
        """,
    )

    p.yaxis.ticker = FixedTicker(ticks=process_ticks + 0.1)
    p.yaxis.major_tick_line_color = None

    p.toolbar.active_scroll = p.select(dict(type=WheelZoomTool))[0]
    p.on_event(RangesUpdate, update_cds)

    # Move legend to the right
    p.add_layout(p.legend[0], "below")
    p.legend.orientation = "horizontal"
    p.legend.location = "center"

    # Make initial call to our callback
    update_cds(None)

    # Hover config
    hover = p.select(HoverTool)
    hover.tooltips = get_html_tooltips(
        {
            "Group Name": "@group_name",
            "Column Name": "@column_name",
            # "Process": "@Process",
            # "Start Time": "@{start_time}{custom} [@{index}]",
            # "End Time": "@{end_time}{custom} [@{_matching_event}]",
            "Time": "@{total_time}{custom}",
            # "Time (Exc)": "@{time.exc}{custom}",
        }
    )
    hover.formatters = {
        # "@{start_time}": get_time_hover_formatter(),
        # "@{end_time}": get_time_hover_formatter(),
        "@{total_time}": get_time_hover_formatter(),
        # "@{time.exc}": get_time_hover_formatter(),
    }
    hover.renderers = [hbar, scatter] if instant_events else [hbar]
    hover.callback = CustomJS(
        code="""
        let hbar_tooltip = document.querySelector('.bk-tooltip');
        let scatter_tooltip = hbar_tooltip.nextElementSibling;

        if (hbar_tooltip && scatter_tooltip &&
            hbar_tooltip.style.display != 'none' &&
            scatter_tooltip.style.display != 'none')
        {
            hbar_tooltip.style.display = 'none';
        }
    """
    )

    # # Range selector
    # profile = trace.time_profile(num_bins=100).fillna(0)
    # func = profile.columns[3:]
    # select = figure(
    #     tools="",
    #     height=80,
    #     title="Drag the middle and edges of the selection box"
    #     "to change the range above",
    #     # title="Timeline",
    #     toolbar_location=None,
    #     y_axis_type=None,
    #     x_range=(min_ts, max_ts + (max_ts - min_ts) * 0.05),
    #     # x_axis_location="above",
    # )
    # bin_size = profile.loc[0]["bin_start"] - profile.loc[0]["bin_end"]
    # palette = get_palette(trace)
    # select.vbar_stack(
    #     func,
    #     x=dodge("bin_start", -bin_size / 2, p.x_range),
    #     width=bin_size,
    #     color=[palette[f] for f in func],
    #     source=profile,
    #     fill_alpha=1.0,
    #     line_width=1,
    # )
    # select.xaxis.formatter = get_time_tick_formatter()

    # range_tool = RangeTool(x_range=p.x_range)
    # range_tool.overlay.fill_color = "navy"
    # range_tool.overlay.fill_alpha = 0.35

    # select.add_tools(range_tool)
    # select.toolbar.active_multi = range_tool

    # Return plot with wrapper function
    return plot(column(p, sizing_mode="stretch_width"))
    # return plot(column(p, select, sizing_mode="stretch_width"))


def operation_histogram(gf: GraphFrame, group_by_column: str = 'name', compared_column: [str] = ['time (inc)']):
    """Displays the result of Trace.message_histogram as a bar graph

    The heights of the bars represent the frequency of messages per size range."""
    # hist, edges = trace.message_histogram(**kwargs)

    # define number of functions to look at
    num_comp = 5

    # FIRST: Get Flat Profile
    flat_profile: pd.DataFrame = gf.flat_profile(group_by_column).head(num_comp)
    flat_profile = flat_profile[compared_column]
    max_val = flat_profile.max(numeric_only=True).max()

    # SECOND: Get the column to compare on
    groups = flat_profile.index.values.tolist()

    data = {group_by_column : groups}

    
    for comp_col in compared_column:
        data[comp_col] = flat_profile[comp_col].to_list()

    source = ColumnDataSource(data=data)

    colors = Accent8[0:len(compared_column)]

    p = figure(x_range=groups, y_range=(0, int(max_val * 1.3)), title="Operation Histogram",
            height=350, toolbar_location=None, tools="")

    for index, comp_col in enumerate(compared_column):
        offset = 0
        if index == 0:
            offset = -0.125
        elif index == len(compared_column) - 1:
            offset = 0.125
        else :
            offset = 0.0625
        p.vbar(x=dodge(group_by_column, offset, range=p.x_range), top=comp_col, source=source,
            width=0.2, color=colors[index], legend_label=comp_col, name=comp_col)

    # p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = pi/4

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    p.add_tools(HoverTool(
        tooltips=[
            # specify the group
            (group_by_column, "@" + group_by_column),
            # specifeis the column in the group
            ('column name', "$name"),
            # specifies the value of the column
            ('value', "@$name")
        ]
    ))


    return plot(p)

def profile_timeline(gf: GraphFrame, group_by_column: str = 'name', compared_column: [str] = ['time (inc)']):
    # compared column is going to act like the process column
    # group_by_column is going to act like the name column 
    # so we are going to create horizontal bars like so:
    # comp_col_0: |--group_by_column[0]--||--group_by_column[1]--|
    # comp_col_1: |--group_by_column[0]--||--group_by_column[1]--|

    # so fore each comp_col, we will construct enter and leave events for each value in the 
    # compare_on_column and then plot them on the timeline

    # so our df will have the following columns:
    # column_name, operation_name, total_time, start_time, end_time, depth

    # trace_df = pd.DataFrame(columns=['column_name', 'group_name', 'total_time', 'start_time', 'end_time', 'depth'])

    flat_profile: pd.DataFrame = gf.flat_profile(groupby_column=group_by_column).sort_values(by=compared_column[0], ascending=False)
    flat_profile = flat_profile[compared_column]

    # get all compare_on
    groups = flat_profile.index.values.tolist()
    
    # setup
    last_end_time = 0
    column_names = []
    group_names = []
    total_times = []
    start_times = []
    end_times = []
    depths = []

    # iterate through each group
    for group in groups:
        df_row = flat_profile.loc[group]
        max_time = df_row.max()
        # print(df_row)
        # iterate through each compared column
        for col in compared_column:
            # print(col, df_row[col])
            column_names.append(col)
            group_names.append(group)
            total_times.append(df_row[col])
            start_times.append(last_end_time)
            end_times.append(last_end_time + df_row[col])
            depths.append(0)
            if df_row[col] < max_time:
                # add idle time to compensate for the difference
                column_names.append(col)
                group_names.append('Idle')
                total_times.append(max_time - df_row[col])
                start_times.append(last_end_time + df_row[col])
                end_times.append(last_end_time + max_time)
                depths.append(0)
        last_end_time += max_time
        # print(df_row.index.to_list())
        # print(df_row.max())
        # print(type(df_row))
    # add row for idle time
    # for col in compared_column:
    #     column_names.insert(0, col)
    #     group_names.insert(0, 'Idle')
    #     total_times.insert(0, last_end_time)
    #     start_times.insert(0, 0)
    #     end_times.insert(0, last_end_time)
    #     depths.insert(0, 0)
    trace_df = pd.DataFrame({
        'column_name': column_names,
        'group_name': group_names,
        'total_time': total_times,
        'start_time': start_times,
        'end_time': end_times,
        '_depth': depths
    })
    return timeline(trace_df, 0, last_end_time, show_depth=True)

        
            

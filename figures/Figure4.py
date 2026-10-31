# -*- coding: utf-8 -*-

"""Seasonality and water extremes, Figure 4."""


from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
)

from matplotlib.ticker import FuncFormatter


ROOT = Path(
    r"K:\Monthly surface water dynamics TIB"
    r"\supervised classification\step4_analysis"
    r"\QTP_water_area_25km_QTP_Albers"
)


SEASONALITY_SHP = (
    ROOT
    / "seasonality_grid"
    / "QTP_25km_seasonality_centres.shp"
)


CHANGE_SHP = (
    ROOT
    / "seasonality_change"
    / "QTP_25km_seasonality_change_centres_OLS.shp"
)


REGIONAL_CSV = (
    ROOT
    / "seasonality_final_OLS"
    / "Regional_seasonality_metrics.csv"
)


QTP_SHP = Path(
    r"D:\phdl2\QTP surface water dynamics"
    r"\regions\QTP.shp"
)


OUTPUT_DIR = (
    ROOT
    / "Figures"
)


PNG_OUT = (
    OUTPUT_DIR
    / "QTP_seasonality_AB_final.png"
)


PDF_OUT = (
    OUTPUT_DIR
    / "QTP_seasonality_AB_final.pdf"
)


P_THRESHOLD = 0.05


GRID_SIZE = 7.2


SIG_SIZE = 1.8


BOUNDARY_WIDTH = 0.80


# 3. COLOUR SCALE


A_VMIN = 0.0
A_VCENTER = 50.0
A_VMAX = 100.0


B_VMIN = -100.0
B_VCENTER = 0.0
B_VMAX = 100.0


CB_FRACTION = 0.035
CB_PAD = 0.020
CB_SHRINK = 0.82
CB_ASPECT = 25


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 8,

        "axes.linewidth": 0.7,

        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CMAP_A = LinearSegmentedColormap.from_list(
    "seasonality_contribution",
    [
        "#9e2f2f",
        "#c94741",
        "#df7968",
        "#efb3a5",
        "#f4d9d3",

        "#f7f7f7",

        "#dce9f1",
        "#abcfe1",
        "#73afd0",
        "#3c8abd",
        "#2166ac",
    ],
    N=256,
)


CMAP_B = LinearSegmentedColormap.from_list(
    "seasonality_change",
    [
        "#8c510a",
        "#a96b1e",
        "#c58a3a",
        "#d8ae70",
        "#ead7b5",

        "#f7f7f7",

        "#dedcea",
        "#beb8d8",
        "#988fc0",
        "#7465a7",
        "#542788",
    ],
    N=256,
)


A_RED = CMAP_A(0.08)
A_BLUE = CMAP_A(0.92)

B_BROWN = CMAP_B(0.08)
B_PURPLE = CMAP_B(0.92)


def read_data():

    print()
    print("=" * 80)
    print("READING EXISTING RESULTS")
    print("=" * 80)


    for path in [
        SEASONALITY_SHP,
        CHANGE_SHP,
        QTP_SHP,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nCannot find:\n{path}"
            )


    seasonality = gpd.read_file(
        SEASONALITY_SHP
    )

    change = gpd.read_file(
        CHANGE_SHP
    )

    qtp = gpd.read_file(
        QTP_SHP
    )


    if "Cseason" not in seasonality.columns:

        raise KeyError(
            "\nPanel a requires field 'Cseason'.\n\n"
            f"Available fields:\n"
            f"{list(seasonality.columns)}"
        )


    required_b = {
        "RelDec",
        "OLSp",
    }


    missing_b = (
        required_b
        -
        set(change.columns)
    )


    if missing_b:

        raise KeyError(
            "\nPanel b requires RelDec and OLSp.\n\n"
            f"Missing fields:\n"
            f"{sorted(missing_b)}\n\n"
            f"File:\n"
            f"{CHANGE_SHP}\n\n"
            f"Available fields:\n"
            f"{list(change.columns)}"
        )


    seasonality["Cseason"] = pd.to_numeric(
        seasonality["Cseason"],
        errors="coerce"
    )


    change["RelDec"] = pd.to_numeric(
        change["RelDec"],
        errors="coerce"
    )


    change["OLSp"] = pd.to_numeric(
        change["OLSp"],
        errors="coerce"
    )


    seasonality = seasonality[
        np.isfinite(
            seasonality["Cseason"]
        )
    ].copy()


    change = change[
        np.isfinite(
            change["RelDec"]
        )
    ].copy()


    # CRS CHECK


    if seasonality.crs is None:

        raise ValueError(
            "Panel a has no CRS."
        )


    if change.crs is None:

        raise ValueError(
            "Panel b has no CRS."
        )


    if qtp.crs is None:

        raise ValueError(
            "QTP boundary has no CRS."
        )


    print()
    print("Panel a CRS:")
    print(seasonality.crs)


    print()
    print("Panel b CRS:")
    print(change.crs)


    if seasonality.crs != change.crs:

        raise ValueError(
            "\nPanel a and Panel b statistical layers "
            "have different CRS.\n\n"
            "The plotting script will NOT reproject "
            "statistical results."
        )


    qtp_plot = qtp.to_crs(
        seasonality.crs
    )


    print()
    print(
        f"Panel a cells: {len(seasonality):,}"
    )


    print(
        f"Panel b cells: {len(change):,}"
    )


    print()
    print(
        "Statistical layers: NOT reprojected"
    )


    print(
        "QTP boundary: transformed IN MEMORY only"
    )


    print("=" * 80)


    return (
        seasonality,
        change,
        qtp_plot,
    )


def get_qtp_cseason():

    if not REGIONAL_CSV.exists():

        raise FileNotFoundError(
            "\nCannot find regional statistics:\n"
            f"{REGIONAL_CSV}"
        )


    df = pd.read_csv(
        REGIONAL_CSV
    )


    required = {
        "Region",
        "SizeClass",
        "Cseason_pct",
    }


    missing = (
        required
        -
        set(df.columns)
    )


    if missing:

        raise KeyError(
            "\nRegional CSV missing fields:\n"
            f"{sorted(missing)}\n\n"
            f"Available fields:\n"
            f"{list(df.columns)}"
        )


    region = (
        df["Region"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    sizeclass = (
        df["SizeClass"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    row = df[
        (region == "QTP")
        &
        (sizeclass == "total")
    ]


    if len(row) != 1:

        raise ValueError(
            "\nExpected exactly one QTP / Total record.\n"
            f"Found: {len(row)}"
        )


    value = pd.to_numeric(
        row.iloc[0]["Cseason_pct"],
        errors="coerce"
    )


    value = float(
        value
    )


    if (
        not np.isfinite(value)
        or
        value < 0
        or
        value > 100
    ):

        raise ValueError(
            f"Invalid QTP Cseason: {value}"
        )


    return value


def calculate_change_proportions(
    change
):

    values = change[
        "RelDec"
    ].to_numpy(
        dtype=float
    )


    values = values[
        np.isfinite(values)
    ]


    n_inc = int(
        np.sum(
            values > 0
        )
    )


    n_dec = int(
        np.sum(
            values < 0
        )
    )


    n_zero = int(
        np.sum(
            np.isclose(
                values,
                0.0
            )
        )
    )


    denominator = (
        n_inc
        +
        n_dec
    )


    if denominator == 0:

        raise ValueError(
            "No positive/negative RelDec cells."
        )


    inc_pct = (
        n_inc
        /
        denominator
        *
        100.0
    )


    dec_pct = (
        n_dec
        /
        denominator
        *
        100.0
    )


    print()
    print("=" * 60)
    print("PANEL B: DIRECTIONAL AREA")
    print("=" * 60)


    print(
        f"Increasing cells : {n_inc:,}"
    )


    print(
        f"Decreasing cells : {n_dec:,}"
    )


    print(
        f"Zero cells       : {n_zero:,}"
    )


    print()
    print(
        f"Increasing area  : {inc_pct:.2f}%"
    )


    print(
        f"Decreasing area  : {dec_pct:.2f}%"
    )


    print("=" * 60)


    return (
        inc_pct,
        dec_pct,
    )


def get_significant_cells(
    change
):

    p = change[
        "OLSp"
    ].to_numpy(
        dtype=float
    )


    valid = np.isfinite(
        p
    )


    significant_mask = (
        valid
        &
        (p < P_THRESHOLD)
    )


    sig = change.loc[
        significant_mask
    ].copy()


    n_valid = int(
        np.sum(
            valid
        )
    )


    n_sig = int(
        len(sig)
    )


    print()
    print("=" * 60)
    print("PANEL B: OLS SIGNIFICANCE")
    print("=" * 60)


    print(
        f"Valid p-values : {n_valid:,}"
    )


    print(
        f"p < 0.05 cells : {n_sig:,}"
    )


    if n_valid > 0:

        print(
            f"Significant    : "
            f"{n_sig / n_valid * 100:.2f}%"
        )


    print("=" * 60)


    return sig


def get_extent(
    qtp,
    padding=0.018
):

    xmin, ymin, xmax, ymax = (
        qtp.total_bounds
    )


    dx = (
        xmax
        -
        xmin
    )


    dy = (
        ymax
        -
        ymin
    )


    return (
        xmin - dx * padding,
        xmax + dx * padding,

        ymin - dy * padding,
        ymax + dy * padding,
    )


def style_axis(
    ax,
    extent
):

    xmin, xmax, ymin, ymax = extent


    ax.set_xlim(
        xmin,
        xmax
    )


    ax.set_ylim(
        ymin,
        ymax
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.set_xticks([])
    ax.set_yticks([])


    ax.set_xlabel("")
    ax.set_ylabel("")


    for spine in ax.spines.values():

        spine.set_visible(
            False
        )


def draw_qtp_boundary(
    ax,
    qtp
):

    qtp.boundary.plot(
        ax=ax,

        color="black",

        linewidth=BOUNDARY_WIDTH,

        zorder=20,
    )


def add_panel_letter(
    ax,
    text
):

    ax.text(
        -0.018,
        1.018,

        text,

        transform=ax.transAxes,

        fontsize=14,

        fontweight="bold",

        ha="right",
        va="bottom",

        zorder=50,
    )


def add_pie_a(
    ax,
    cseason
):

    seasonal = float(
        cseason
    )


    nonseasonal = (
        100.0
        -
        seasonal
    )


    pie_ax = ax.inset_axes(
        [
            -0.070,
            -0.075,
            0.365,
            0.365,
        ]
    )


    wedges, _, autotexts = pie_ax.pie(
        [
            seasonal,
            nonseasonal,
        ],

        colors=[
            A_BLUE,
            A_RED,
        ],

        startangle=90,

        counterclock=False,

        radius=1.0,

        autopct=lambda p:
            f"{p:.1f}%",

        pctdistance=0.62,

        wedgeprops={
            "edgecolor": "white",
            "linewidth": 0.7,
        },

        textprops={
            "fontsize": 8.2,
        },
    )


    for text in autotexts:

        text.set_fontsize(
            8.2
        )

        text.set_color(
            "black"
        )


    pie_ax.legend(
        wedges,

        [
            "Seasonality",
            "Non-seasonality",
        ],

        loc="upper left",

        bbox_to_anchor=(
            0.88,
            0.42,
        ),

        frameon=False,

        fontsize=7.7,

        handlelength=1.05,

        handletextpad=0.45,

        labelspacing=0.45,

        borderaxespad=0.0,
    )


    pie_ax.set_aspect(
        "equal"
    )


    pie_ax.set_axis_off()


def add_pie_b(
    ax,
    inc_pct,
    dec_pct
):

    pie_ax = ax.inset_axes(
        [
            -0.070,
            -0.075,
            0.365,
            0.365,
        ]
    )


    wedges, _, autotexts = pie_ax.pie(
        [
            inc_pct,
            dec_pct,
        ],

        colors=[
            B_PURPLE,
            B_BROWN,
        ],

        startangle=90,

        counterclock=False,

        radius=1.0,

        autopct=lambda p:
            f"{p:.1f}%",

        pctdistance=0.62,

        wedgeprops={
            "edgecolor": "white",
            "linewidth": 0.7,
        },

        textprops={
            "fontsize": 8.2,
        },
    )


    for text in autotexts:

        text.set_fontsize(
            8.2
        )

        text.set_color(
            "black"
        )


    pie_ax.legend(
        wedges,

        [
            "Increasing",
            "Decreasing",
        ],

        loc="upper left",

        bbox_to_anchor=(
            0.88,
            0.42,
        ),

        frameon=False,

        fontsize=7.7,

        handlelength=1.05,

        handletextpad=0.45,

        labelspacing=0.45,

        borderaxespad=0.0,
    )


    pie_ax.set_aspect(
        "equal"
    )


    pie_ax.set_axis_off()


def draw_panel_a(
    ax,
    seasonality,
    qtp,
    extent,
    qtp_cseason
):

    values = seasonality[
        "Cseason"
    ].to_numpy(
        dtype=float
    )


    norm = TwoSlopeNorm(
        vmin=A_VMIN,
        vcenter=A_VCENTER,
        vmax=A_VMAX
    )


    scatter = ax.scatter(
        seasonality.geometry.x,
        seasonality.geometry.y,

        c=values,

        cmap=CMAP_A,

        norm=norm,

        marker="s",

        s=GRID_SIZE,

        linewidths=0,

        edgecolors="none",

        antialiased=False,

        zorder=5,

        rasterized=False,
    )


    draw_qtp_boundary(
        ax,
        qtp
    )


    style_axis(
        ax,
        extent
    )


    cb = plt.colorbar(
        scatter,

        ax=ax,

        orientation="vertical",

        fraction=CB_FRACTION,

        pad=CB_PAD,

        shrink=CB_SHRINK,

        aspect=CB_ASPECT,
    )


    cb.set_ticks(
        [
            0,
            25,
            50,
            75,
            100,
        ]
    )


    cb.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos:
                f"{int(round(x))}"
        )
    )


    cb.ax.tick_params(
        labelsize=7.5,

        length=2.5,

        pad=2,
    )


    cb.set_label(
        "Relative contribution of seasonality (%)",

        fontsize=8.2,

        labelpad=7,
    )


    cb.outline.set_linewidth(
        0.6
    )


    cb.ax.axhline(
        A_VCENTER,

        color="0.25",

        linewidth=0.65,

        linestyle="--"
    )


    add_panel_letter(
        ax,
        "a"
    )


    add_pie_a(
        ax,
        qtp_cseason
    )


def draw_panel_b(
    ax,
    change,
    significant,
    qtp,
    extent,
    inc_pct,
    dec_pct
):

    values = change[
        "RelDec"
    ].to_numpy(
        dtype=float
    )


    display_values = np.clip(
        values,
        B_VMIN,
        B_VMAX
    )


    norm = TwoSlopeNorm(
        vmin=B_VMIN,
        vcenter=B_VCENTER,
        vmax=B_VMAX
    )


    scatter = ax.scatter(
        change.geometry.x,
        change.geometry.y,

        c=display_values,

        cmap=CMAP_B,

        norm=norm,

        marker="s",

        s=GRID_SIZE,

        linewidths=0,

        edgecolors="none",

        antialiased=False,

        zorder=5,

        rasterized=False,
    )


    if len(significant) > 0:

        ax.scatter(
            significant.geometry.x,
            significant.geometry.y,

            s=SIG_SIZE,

            marker="o",

            c="black",

            edgecolors="none",

            linewidths=0,

            alpha=0.90,

            zorder=30,

            rasterized=False,
        )


    draw_qtp_boundary(
        ax,
        qtp
    )


    style_axis(
        ax,
        extent
    )


    cb = plt.colorbar(
        scatter,

        ax=ax,

        orientation="vertical",

        fraction=CB_FRACTION,

        pad=CB_PAD,

        shrink=CB_SHRINK,

        aspect=CB_ASPECT,

        extend="both",
    )


    cb.set_ticks(
        [
            -100,
            -50,
            0,
            50,
            100,
        ]
    )


    cb.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos:
                f"{int(round(x))}"
        )
    )


    cb.ax.tick_params(
        labelsize=7.5,

        length=2.5,

        pad=2,
    )


    cb.set_label(
        "Relative change in seasonality\n(% per decade)",

        fontsize=8.2,

        labelpad=7,
    )


    cb.outline.set_linewidth(
        0.6
    )


    cb.ax.axhline(
        B_VCENTER,

        color="0.25",

        linewidth=0.65,

        linestyle="--"
    )


    add_panel_letter(
        ax,
        "b"
    )


    add_trend_histogram(ax, change)


BASE = ROOT.parent
QTP_MONTHLY = BASE / 'QTP_waterbody_statistics_0p1' / 'QTP_monthly_waterbody_statistics.csv'
WATERSHED_MONTHLY = BASE / 'QTP_waterbody_statistics_0p1' / 'Watershed_monthly_waterbody_statistics.csv'

COLORS = {'Total':'#858585','Small':'#83C6A0','Medium':'#9DD6E5','Large':'#FAD793'}
AREA_COLS = {'Total':'Total_surface_water_area_km2','Small':'Small_waterbody_area_km2',
             'Medium':'Medium_waterbody_area_km2','Large':'Large_waterbody_area_km2'}

HIST_BINS_PER_SIDE = 18
HIST_RECT = [0.015, -0.060, 0.34, 0.30]


def add_trend_histogram(ax, change):
    values=change.RelDec.to_numpy(float)
    values=values[np.isfinite(values)]
    if not len(values):raise ValueError('No finite grid trends')
    inset=ax.inset_axes(HIST_RECT,zorder=45)

    limit=300.0
    bins=np.linspace(-limit,limit,2*HIST_BINS_PER_SIDE+1)
    counts,_=np.histogram(values,bins=bins)
    centres=(bins[:-1]+bins[1:])/2
    inset.bar(centres,counts,width=np.diff(bins)*.97,
              color=[B_BROWN if x<0 else B_PURPLE for x in centres],linewidth=0)
    inset.axvline(0,color='.35',lw=.55,ls='--')
    inset.set_xlim(-limit,limit)
    inset.set_xticks([-300,-150,0,150,300])
    inset.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0f}' if limit>=1 else f'{x:.2g}'))
    inset.tick_params(labelsize=6,length=2,pad=1)
    inset.set_xlabel('Relative change (% per decade)',fontsize=6.5,labelpad=1)
    inset.set_ylabel('Grid cells',fontsize=6.5,labelpad=1)
    inset.spines[['top','right']].set_visible(False)
    inset.spines[['bottom','left']].set_linewidth(.5)
    inset.yaxis.set_major_locator(plt.MaxNLocator(3,integer=True))
    inset.set_facecolor('none')
    inset.patch.set_alpha(0)


def check_months(frame, columns):
    frame=frame.copy()
    frame['Date']=pd.to_datetime(dict(year=frame.Year,month=frame.Month,day=1))
    frame=frame.sort_values('Date')
    if frame.Date.duplicated().any() or not pd.DatetimeIndex(frame.Date).equals(pd.date_range('2016-01-01','2025-12-01',freq='MS')):
        raise ValueError('Monthly series must contain exactly 120 months, 2016–2025')
    for col in columns:
        frame[col]=pd.to_numeric(frame[col],errors='raise')
        if not np.isfinite(frame[col]).all() or (frame[col]<0).any():raise ValueError(f'Invalid area: {col}')
    return frame


def style_chart(ax, letter, title):
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(title,loc='left',fontsize=10,pad=10)
    ax.text(-.10,1.045,letter,transform=ax.transAxes,fontsize=14,fontweight='bold',ha='left')
    ax.tick_params(labelsize=8)


SIZE_CONTRIBUTION_CSV = BASE / 'QTP_seasonality_results' / 'Size_class_seasonal_contribution.csv'
A_INSET_RECT = [0.040, -0.095, 0.31, 0.30]

def add_pie_a(ax, cseason):

    table=pd.read_csv(REGIONAL_CSV,encoding='utf-8-sig')
    table['Region']=table.Region.astype(str).str.strip().str.upper()
    table['SizeClass']=table.SizeClass.astype(str).str.strip().str.capitalize()
    rows=table[table.Region=='QTP'].set_index('SizeClass')
    names=['Small','Medium','Large','Total']
    if rows.index.duplicated().any() or not set(names).issubset(rows.index):
        raise ValueError('Expected one QTP record for each of Total/Small/Medium/Large')
    values=pd.to_numeric(rows.loc[names,'Cseason_pct'],errors='raise').to_numpy()
    if not np.isfinite(values).all() or ((values<0)|(values>100)).any():raise ValueError('Invalid Cseason_pct')
    inset=ax.inset_axes(A_INSET_RECT,zorder=45)
    inset.patch.set_alpha(0)
    inset.bar(range(4),values,color=[COLORS[n] for n in names],width=.42,edgecolor='.25',linewidth=.45)
    inset.axhline(50,color='.3',lw=.6,ls='--')
    inset.set_ylim(0,105);inset.set_yticks([0,50,100])
    inset.set_xticks(range(4),names)
    inset.tick_params(labelsize=6.2,length=2,pad=1)
    inset.set_ylabel('Seasonality contribution (%)',fontsize=6.3,labelpad=1)
    for i,v in enumerate(values):inset.text(i,v+3,f'{v:.1f}',ha='center',fontsize=6.2)
    inset.spines[['top','right']].set_visible(False)
    inset.spines[['left','bottom']].set_linewidth(.5)


def read_size_contributions():
    path=SIZE_CONTRIBUTION_CSV
    if not path.exists():
        alternate=path.with_name('Size_class_seasonal_contribution(1).csv')
        if alternate.exists():path=alternate
        else:raise FileNotFoundError(f'Missing covariance contribution table: {path}')
    table=pd.read_csv(path,encoding='utf-8-sig')
    required={'Region','SizeClass','SeasonalContribution_pct'}
    if not required.issubset(table.columns):raise ValueError(f'Missing fields: {required-set(table.columns)}')
    table['Region']=table.Region.astype(str).str.strip()
    table['SizeClass']=table.SizeClass.astype(str).str.strip().str.capitalize()
    if table.duplicated(['Region','SizeClass']).any():raise ValueError('Duplicate covariance contribution records')
    table['SeasonalContribution_pct']=pd.to_numeric(table.SeasonalContribution_pct,errors='raise')
    if not np.isfinite(table.SeasonalContribution_pct).all():raise ValueError('Undefined covariance contributions')
    for region,group in table.groupby('Region'):
        if set(group.SizeClass)!={'Small','Medium','Large'}:raise ValueError(f'Incomplete classes: {region}')
        if not np.isclose(group.SeasonalContribution_pct.sum(),100,atol=1e-5,rtol=0):raise ValueError(f'Contributions do not sum to 100: {region}')
    if 'QTP' not in set(table.Region):raise ValueError('No QTP covariance contribution rows')
    return table


def draw_contribution_heatmap(ax, table, fig):
    """Every column refers to that region's own total seasonal variance.

    Negative covariance contributions are retained. No renormalisation.
    """
    regions=['QTP','KlB','NrB','QdB','QlB','QtB','STB','SWB','YB','YtB']
    classes=['Small','Medium','Large']
    frame=table.pivot(index='SizeClass',columns='Region',values='SeasonalContribution_pct')
    frame=frame.loc[classes,regions]
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError('Missing region/class covariance contributions')
    values=frame.to_numpy(float)

    cmap=LinearSegmentedColormap.from_list('contribution', ['#E8878B','#FAFAF7','#83C6A0','#408B68'])
    # Set white exactly at zero and retain a symmetric quantitative scale.
    cmap=LinearSegmentedColormap.from_list('contribution',
        [(0,'#E8878B'),(.5,'#FAFAF7'),(.8,'#83C6A0'),(1,'#408B68')])
    limit=max(100.0,float(np.ceil(np.abs(values).max()/10)*10))
    im=ax.imshow(values,aspect='auto',cmap=cmap,
                 norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    style_chart(ax,'d','Size-class contributions to regional seasonal variance')
    ax.set_xticks(range(len(regions)),regions)
    ax.set_yticks(range(3),classes)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(len(regions)):
            value=values[i,j]
            ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=9,
                    fontweight='bold' if j==0 else 'normal',
                    color='white' if value>limit*.85 else '#252525')
    ax.axvline(.5,color='.3',lw=1)
    for label in ax.get_xticklabels():
        if label.get_text()=='QTP':label.set_fontweight('bold')
    cb=fig.colorbar(im,ax=ax,pad=.015,fraction=.025,aspect=20)
    cb.set_label('Covariance contribution (%)',fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return frame


def draw_regional_summary(ax, regional_table):
    """Summarise the two map metrics for total surface water by region."""
    regions=['QTP','KlB','NrB','QdB','QlB','QtB','STB','SWB','YB','YtB']
    total=regional_table[regional_table.SizeClass.eq('Total')].copy()
    if total.duplicated('Region').any():
        raise ValueError('Duplicate regional Total records')
    total=total.set_index('Region').loc[regions]
    x=pd.to_numeric(total.Cseason_pct,errors='raise').to_numpy(float)
    y=pd.to_numeric(total.RelDec_pct_per_decade,errors='raise').to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing regional Total metrics')

    # Light quadrant shading: seasonal/non-seasonal dominance × increase/decrease.
    xmin=max(0.0,float(np.floor((x.min()-5)/10)*10))
    xmax=min(100.0,float(np.ceil((x.max()+5)/10)*10))
    ymax=max(5.0,float(np.ceil(np.max(np.abs(y))*1.18/5)*5))
    ax.axvspan(xmin,50,color='#E8878B',alpha=.075,zorder=0)
    ax.axvspan(50,xmax,color='#83C6A0',alpha=.075,zorder=0)
    ax.axhline(0,color='.35',lw=.75,ls='--',zorder=1)
    ax.axvline(50,color='.35',lw=.75,ls='--',zorder=1)
    ax.scatter(x[1:],y[1:],s=42,color='#95ACD2',edgecolor='.22',linewidth=.55,zorder=3)
    ax.scatter(x[0],y[0],s=105,marker='*',color='#E8878B',edgecolor='.18',linewidth=.65,zorder=4)
    for i,region in enumerate(regions):
        offset=(5,5) if y[i]>=0 else (5,-9)
        ax.annotate(region,(x[i],y[i]),xytext=offset,textcoords='offset points',
                    fontsize=7.5,fontweight='bold' if region=='QTP' else 'normal')
    ax.set_xlim(xmin,xmax);ax.set_ylim(-ymax,ymax)
    ax.set_xlabel('Seasonality contribution, $C_{seasonality}$ (%)')
    ax.set_ylabel('Change in seasonal magnitude (% per decade)')
    style_chart(ax,'c','Regional seasonality dominance and change')


def draw_contribution_option(ax, table, fig, option):
    """Same seasonal-variance attribution in three visual encodings.

    Heatmaps use a LINEAR scale, normally -10 to 100, so small negative
    values do not occupy half the colourbar. Zero is explicitly near-white.
    Bounds expand if future inputs exceed these limits; no values are clipped.
    """
    from matplotlib.colors import Normalize
    regions=['QTP','KlB','QdB','QlB','NrB','QtB','YtB','SWB','YB','STB']
    table=table.copy()
    table['Region']=table['Region'].replace({'QIB':'QlB'})
    region_labels=['QTP','KlB','QdB','QIB','NrB','QtB','YtB','SWB','YB','STB']
    classes=['Small','Medium','Large']
    frame=table.pivot(index='SizeClass',columns='Region',values='SeasonalContribution_pct').loc[classes,regions]
    values=frame.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Missing regional size-class contribution')
    lo=min(-10.,float(np.floor(values.min()/5)*5))
    hi=max(100.,float(np.ceil(values.max()/10)*10))
    title='Contributions to regional seasonal variability'
    style_chart(ax,'d',title)
    if option=='bars':

        yy=np.arange(len(regions));height=.23
        for i,name in enumerate(classes):
            ax.barh(yy+(i-1)*height,values[i],height=height,
                    color=COLORS[name],edgecolor='.35',linewidth=.35,label=name)
        ax.axvline(0,color='.25',lw=.65)
        ax.set_yticks(yy,region_labels);ax.invert_yaxis()
        ax.set_xlim(lo,hi);ax.set_xticks([lo,0,25,50,75,100])
        ax.set_xlabel('Contribution to seasonal variability (%)',fontsize=8)
        ax.legend(frameon=False,ncol=3,fontsize=7,loc='lower right')
        ax.tick_params(axis='y',length=0,labelsize=7)
        ax.grid(axis='x',alpha=.15,lw=.5);ax.set_axisbelow(True)
    else:
        zero=(0-lo)/(hi-lo)
        cmap=LinearSegmentedColormap.from_list('muted_teal_contribution',
            [(0,'#D6A0A0'),(zero,'#FAFAF7'),
             (zero+(1-zero)*.45,'#B7DAD3'),(1,'#397D78')])
        transposed=option=='heatmap_transposed'
        shown=values.T if transposed else values
        im=ax.pcolormesh(np.arange(shown.shape[1]+1)-.5,
                         np.arange(shown.shape[0]+1)-.5,shown,
                         cmap=cmap,norm=Normalize(lo,hi),shading='flat',rasterized=False)
        ax.set_xlim(-.5,shown.shape[1]-.5)
        ax.set_ylim(shown.shape[0]-.5,-.5)
        ax.set_xticks(range(shown.shape[1]),classes if transposed else region_labels)
        ax.set_yticks(range(shown.shape[0]),region_labels if transposed else classes)
        ax.tick_params(length=0,labelsize=7)
        for i in range(shown.shape[0]):
            for j in range(shown.shape[1]):
                v=shown[i,j]
                ax.text(j,i,f'{v:.1f}',ha='center',va='center',
                        fontsize=7.2 if transposed else 8,
                        fontweight='bold' if (i==0 if transposed else j==0) else 'normal',
                        color='white' if v>75 else '#253D3A')
        if transposed:ax.axhline(.5,color='.4',lw=.7)
        else:ax.axvline(.5,color='.4',lw=.7)
        cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.035,aspect=18)
        cb.set_ticks([lo,0,25,50,75,100] if hi==100 else [lo,0,hi/2,hi])
        cb.set_label('Contribution to seasonal variability (%)',fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return frame


# https://github.com/Liluoq/global-dominance-of-seasonality-in-shaping-lake-surface-extent-dynamics/blob/main/my_spatial_analyze/attach_geometry_and_generate_grid.py
# rao_mk_test_on_stl_linear_trend -> hamed_rao_modification_test(trend_terms).
# Use the existing robust STL components from this project's extreme analysis.
# HR variance correction follows pyMannKendall v1.4.3 (the signed sni formula,


# Invalid corrected variances stay undefined; never silently use ordinary MK.

import argparse
import json
from scipy.stats import norm, rankdata
from matplotlib.colors import ListedColormap

EXTREME_DIR = None
EXTREME_ALPHA = 0.05
EXTREME_DPI = 300


JOINT_COLOURS = ['#F0CE8B', '#8DA5C8', '#DA8D96', '#87C39D']
LOW_COLOUR = '#BE727E'
HIGH_COLOUR = '#559CAF'
SIZE_COLOURS = {'Total':'#D1B478','Small':'#87C39D','Medium':'#91CADA','Large':'#8DA5C8'}
EXPECTED_MONTHS = pd.date_range('2016-01-01', '2025-12-01', freq='MS')

def hr_mk_test(values, alpha=EXTREME_ALPHA):
    """Hamed–Rao MK on monthly STL trend; all significant ACF lags, lag=None.

    Equivalent p/z/tau/variance to pyMannKendall 1.4.3 on finite nondegenerate
    series. Constant series return p=1. Nonpositive corrected variance -> NaN.
    p is two-sided and unadjusted for multiple spatial tests, as in the Li map.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        return dict(mk_p=np.nan, mk_z=np.nan, mk_tau=np.nan, mk_var=np.nan, mk_flag='INVALID')
    n = len(x)
    ii, jj = np.triu_indices(n, 1)
    differences = x[jj] - x[ii]
    score = float(np.sign(differences).sum())
    tau = score / (n * (n - 1) / 2)
    _, ties = np.unique(x, return_counts=True)
    variance = (n*(n-1)*(2*n+5) - np.sum(ties*(ties-1)*(2*ties+5))) / 18
    if variance == 0:
        return dict(mk_p=1., mk_z=0., mk_tau=0., mk_var=0., mk_flag='CONSTANT')
    sen = float(np.median(differences/(jj-ii)))
    ranks = rankdata(x - np.arange(1,n+1)*sen)
    centred = ranks-ranks.mean()
    denominator = float(np.dot(centred,centred))

    if denominator > 0:
        acf = np.correlate(centred,centred,'full')[n-1:]/denominator
        cutoff = norm.ppf(1-alpha/2)/np.sqrt(n)
        correction = 0.
        for lag in range(1,n):
            if abs(acf[lag]) > cutoff:
                correction += (n-lag)*(n-lag-1)*(n-lag-2)*acf[lag]
        variance *= 1 + 2*correction/(n*(n-1)*(n-2))
    if not np.isfinite(variance) or variance <= 0:
        return dict(mk_p=np.nan, mk_z=np.nan, mk_tau=tau, mk_var=variance, mk_flag='BAD_VAR')
    z = (score-np.sign(score))/np.sqrt(variance) if score else 0.
    p = float(2*(1-norm.cdf(abs(z))))
    return dict(mk_p=p, mk_z=float(z), mk_tau=tau, mk_var=float(variance), mk_flag='OK')

def find_extreme_directory(explicit=None):
    required = ['Grid_summary.csv', 'Grid_STL_components.npz',
                'Regional_summary.csv', 'Regional_monthly_components.csv',
                'QTP_25km_extremes_centres.shp', 'QTP_25km_extremes_centres.prj']
    choices = [Path(explicit)] if explicit else sorted(ROOT.parent.glob('QTP_trend_extremes_*'), reverse=True)
    for folder in choices:
        try:
            state = json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if state.get('status') == 'COMPLETE' and all((folder/f).is_file() for f in required):
            return folder
    raise FileNotFoundError('No COMPLETE extreme results with Grid_STL_components.npz. Set EXTREME_DIR or --extreme-dir.')

def add_impact_classes(df):
    df = df.copy()
    df['sig_chg'] = np.select([
        (df.mk_p < EXTREME_ALPHA) & (df.beta_yr < 0),
        (df.mk_p < EXTREME_ALPHA) & (df.beta_yr > 0)], [-1,1], default=0)
    for prefix in ['low','high']:
        impact = df['i_'+prefix]
        dev = df[prefix+'_dev']
        right_sign = (dev < 0) if prefix == 'low' else (dev > 0)
        eligible = (df.sig_chg != 0) & np.isfinite(impact) & np.isfinite(dev) & right_sign

        codes = (df.beta_yr > 0).astype(int) + 2*(impact.abs() > 50).astype(int)
        df[prefix+'_cls'] = np.where(eligible,codes,-1)
        df[prefix+'_ok'] = right_sign & np.isfinite(impact) & np.isfinite(dev)
    return df

def compute_significance(folder):
    grid = pd.read_csv(folder/'Grid_summary.csv')
    if grid.grid_id.duplicated().any(): raise ValueError('Duplicate grid_id in Grid_summary')
    with np.load(folder/'Grid_STL_components.npz',allow_pickle=False) as z:
        ids = z['grid_id'].astype(np.int64)
        dates = z['dates'].astype(str)
        trends = z['trend'].astype(float)
    if not np.array_equal(dates, EXPECTED_MONTHS.strftime('%Y-%m').to_numpy()):
        raise ValueError('NPZ dates do not match 120 months, 2016–2025')
    if trends.shape != (len(ids),120) or len(np.unique(ids)) != len(ids):
        raise ValueError('NPZ dimensions or duplicate IDs')
    complete = grid.loc[grid.n_valid.eq(120),'grid_id'].to_numpy()
    if set(complete) != set(ids): raise ValueError('NPZ and grid summary contain different complete grids')
    slopes = np.array([np.polyfit(np.arange(120)/12,y,1)[0] for y in trends])
    expected = grid.set_index('grid_id').loc[ids,'beta_yr'].to_numpy(float)
    if not np.allclose(slopes,expected,atol=1e-7,rtol=1e-8):
        raise ValueError('NPZ trends and saved beta_yr disagree: do not mix runs')
    rows = []
    for k,(gid,y) in enumerate(zip(ids,trends)):
        rows.append(dict(grid_id=int(gid),**hr_mk_test(y)))
        if (k+1) % 500 == 0: print(f'Hamed–Rao MK: {k+1}/{len(ids)}',flush=True)
    grid = add_impact_classes(grid.merge(pd.DataFrame(rows),on='grid_id',how='left',validate='one_to_one'))
    points = gpd.read_file(folder/'QTP_25km_extremes_centres.shp')
    if points.crs is None or not points.crs.is_projected: raise ValueError('Missing/projected grid CRS')
    if not points.geometry.geom_type.eq('Point').all(): raise ValueError('Expected point geometry')
    if points.grid_id.duplicated().any() or set(points.grid_id) != set(grid.grid_id):
        raise ValueError('Shapefile and CSV IDs differ')

    for field in ['beta_yr','i_low','i_high','low_dev','high_dev']:
        a=points.set_index('grid_id').loc[grid.grid_id,field].to_numpy(float)
        if not np.allclose(a,grid[field],equal_nan=True,atol=1e-7,rtol=1e-8):
            raise ValueError('Shapefile and CSV disagree: '+field)
    new_fields = ['grid_id','mk_p','mk_z','mk_tau','mk_var','mk_flag','sig_chg','low_cls','high_cls','low_ok','high_ok']
    points = points.merge(grid[new_fields],on='grid_id',validate='one_to_one')
    region = pd.read_csv(folder/'Regional_summary.csv')
    monthly = pd.read_csv(folder/'Regional_monthly_components.csv')
    keys = ['Scale','Region','SizeClass']
    if region.duplicated(keys).any(): raise ValueError('Duplicate regional records')
    monthly['Date'] = pd.to_datetime(monthly.Date,format='%Y-%m')
    rows = []
    for key,g in monthly.groupby(keys,sort=True):
        g=g.sort_values('Date')
        if not pd.DatetimeIndex(g.Date).equals(EXPECTED_MONTHS): raise ValueError('Incomplete regional series')
        y=g.Trend_km2.to_numpy(float)
        match=region
        for col,value in zip(keys,key): match=match[match[col]==value]
        if len(match)!=1 or not np.isclose(np.polyfit(np.arange(120)/12,y,1)[0],match.iloc[0].beta_yr,atol=1e-7,rtol=1e-8):
            raise ValueError('Regional trend/summary mismatch')
        rows.append(dict(zip(keys,key),**hr_mk_test(y)))
    if len(rows)!=len(region): raise ValueError('Missing regional components')
    region = add_impact_classes(region.merge(pd.DataFrame(rows),on=keys,validate='one_to_one'))
    return grid,points,region

def significant_impact_counts(mapped):
    rows=[]
    names={('low',-1):'At least doubled contraction',('low',1):'Fully offset expansion',
           ('high',-1):'Fully offset contraction',('high',1):'At least doubled expansion'}
    for prefix in ['low','high']:
        for direction in [-1,1]:
            sub=mapped[mapped.sig_chg==direction]
            raw=sub['i_'+prefix].abs()>50
            supported=raw & sub[prefix+'_ok']
            n=len(sub)
            rows.append(dict(Event=prefix,Direction='Declining' if direction<0 else 'Increasing',
                Interpretation=names[prefix,direction],SignificantGrids=n,
                ValidDirectionalEvents=int(sub[prefix+'_ok'].sum()),
                IndexThresholdCount=int(raw.sum()),DirectionConsistentCount=int(supported.sum()),
                OppositeOrZeroDeviation=int((np.isfinite(sub['i_'+prefix])&~sub[prefix+'_ok']).sum()),
                UndefinedIndex=int(sub['i_'+prefix].isna().sum()),
                Percent=100*supported.sum()/n if n else np.nan,
                RawThresholdPercent=100*raw.sum()/n if n else np.nan))
    return pd.DataFrame(rows)

def four_square_legend(ax,prefix):

    matrix = np.array([[2,3],[0,1]]) if prefix=='low' else np.array([[0,1],[2,3]])
    inset=ax.inset_axes([.055,.075,.175,.245],zorder=60)
    inset.set_facecolor('white')
    inset.pcolormesh([-1,0,1],[-100,-50,0] if prefix=='low' else [0,50,100],
                     matrix,cmap=ListedColormap(JOINT_COLOURS),vmin=-.5,vmax=3.5,
                     shading='flat',rasterized=False)
    inset.set_box_aspect(1)
    inset.axvline(0,color='white',lw=.8)
    inset.axhline(-50 if prefix=='low' else 50,color='white',lw=.8)
    inset.set_xticks([-.5,.5]);inset.set_xticklabels(['−','+'],fontsize=7)
    inset.set_yticks([-100,-50,0] if prefix=='low' else [0,50,100])
    inset.tick_params(axis='y',labelsize=6.3,length=2,pad=1)
    inset.tick_params(axis='x',length=0,pad=1)
    inset.set_xlabel('Long-term trend',fontsize=6.5,labelpad=1)
    inset.set_ylabel('Low-water impact (%)' if prefix=='low' else 'High-water impact (%)',fontsize=6.3,labelpad=1)
    for label in [inset.xaxis.label,inset.yaxis.label,*inset.get_xticklabels(),*inset.get_yticklabels()]:
        label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.5))
    for spine in inset.spines.values(): spine.set_linewidth(.5)

def extreme_map(ax,points,qtp,extent,prefix,letter):
    classes=points[prefix+'_cls'].to_numpy(int)
    colours=np.array(['#dedede']*len(points),dtype=object)
    for code,colour in enumerate(JOINT_COLOURS): colours[classes==code]=colour
    ax.scatter(points.geometry.x,points.geometry.y,c=colours,marker='s',s=GRID_SIZE,
               linewidths=0,edgecolors='none',antialiased=False,rasterized=False,zorder=5)
    draw_qtp_boundary(ax,qtp);style_axis(ax,extent);add_panel_letter(ax,letter)
    four_square_legend(ax,prefix)
    ax.set_title('Low-water extremes' if prefix=='low' else 'High-water extremes',fontsize=10,pad=9)
    ax.text(.98,-.055,'Colour: significant trend and direction-consistent event\nGrey: non-significant, undefined or opposite event direction',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=5.8,zorder=50)

def draw_effect_percentages(ax,counts,letter='e'):
    labels=['Low: double\ncontraction','Low: offset\nexpansion','High: offset\ncontraction','High: double\nexpansion']
    values=counts.Percent.to_numpy(float)
    colours=[JOINT_COLOURS[2],JOINT_COLOURS[3],JOINT_COLOURS[2],JOINT_COLOURS[3]]
    bars=ax.bar(np.arange(4),np.nan_to_num(values),color=colours,width=.6)
    for bar,(_,r) in zip(bars,counts.iterrows()):
        label=(f'{r.Percent:.1f}%\n{int(r.DirectionConsistentCount)}/{int(r.SignificantGrids)}'
               if np.isfinite(r.Percent) else 'NA\nn=0')
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,label,ha='center',va='bottom',fontsize=7.4)
    ax.set_xticks(range(4));ax.set_xticklabels(labels,fontsize=7.5)
    ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Significantly changing grid cells (%)',fontsize=8)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Extreme effects within each significant trend group',fontsize=9,pad=10)
    add_panel_letter(ax,letter)

def draw_regional_impacts(ax,region):
    sub=region[region.SizeClass=='Total'].copy()
    order=['QTP']+sorted(set(sub.Region)-{'QTP'})
    sub=sub.set_index('Region').reindex(order)
    ypos=np.arange(len(sub))
    for col,colour,offset in [('i_low',LOW_COLOUR,-.17),('i_high',HIGH_COLOUR,.17)]:
        valid=np.isfinite(sub[col]);sig=(sub.mk_p<EXTREME_ALPHA)&valid
        ax.scatter(sub.loc[valid,col],ypos[valid]+offset,s=24,facecolors='white',edgecolors=colour,
                   linewidths=.9,label='Low water' if col=='i_low' else 'High water',zorder=3)
        ax.scatter(sub.loc[sig,col],ypos[sig]+offset,s=24,c=colour,edgecolors=colour,linewidths=.9,zorder=4)
    labels=[]
    for name,row in sub.iterrows():
        direction='+' if row.beta_yr>0 else '−' if row.beta_yr<0 else '0'
        labels.append(f'{name} ({direction})')
    ax.set_yticks(ypos);ax.set_yticklabels(labels,fontsize=7.5);ax.invert_yaxis()
    for x in [-50,50]: ax.axvline(x,color='.5',ls='--',lw=.7)
    ax.axvline(0,color='.7',lw=.6)
    ax.set_xlim(-105,105);ax.set_xticks([-100,-50,0,50,100]);ax.set_xlabel('Extreme impact (%)')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,fontsize=7,loc='lower right')
    ax.set_title('Total area: QTP and watersheds\nFilled: significant trend; open: non-significant/undefined',fontsize=8.5,pad=10)
    add_panel_letter(ax,'e')

def draw_size_impacts(ax,region,prefix,letter):
    """Nine watershed aggregate series per class; QTP is separate, not a tenth basin.

    No lake-level violin: these are regional sums of contemporaneous size classes.
    Filled markers denote a significant area trend, not significance of the index.
    """
    from matplotlib.lines import Line2D
    classes=['Total','Small','Medium','Large']
    field='i_'+prefix
    for j,size in enumerate(classes):
        basins=region[(region.Scale=='Watershed')&(region.SizeClass==size)].sort_values('Region')
        basins=basins[np.isfinite(basins[field])]
        n=len(basins)
        positions=j+np.linspace(-.18,.18,n) if n>1 else np.array([j])
        for xpos,(_,r) in zip(positions,basins.iterrows()):
            ax.scatter(xpos,r[field],s=23,marker='o',
                facecolors=SIZE_COLOURS[size] if r.mk_p<EXTREME_ALPHA else 'white',
                edgecolors='#667278',linewidths=.7,zorder=3)
        if n:
            median=float(basins[field].median())
            ax.plot([j-.24,j+.24],[median,median],color='#657279',lw=1.0,zorder=2)
        whole=region[(region.Scale=='QTP')&(region.SizeClass==size)]
        if len(whole)!=1: raise ValueError('Expected one QTP row per size class')
        r=whole.iloc[0]
        if np.isfinite(r[field]):
            ax.scatter(j+.29,r[field],s=43,marker='D',
                facecolors='#343B40' if r.mk_p<EXTREME_ALPHA else 'white',
                edgecolors='#343B40',linewidths=1,zorder=5)
            ax.annotate(f'{r[field]:.1f}',(j+.29,r[field]),xytext=(3,4),
                        textcoords='offset points',fontsize=6.8,color='#343B40')
    threshold=-50 if prefix=='low' else 50
    ax.axhline(threshold,color='#777777',ls='--',lw=.7,zorder=1)
    ax.set_xlim(-.45,3.8)
    ax.set_ylim((-105,5) if prefix=='low' else (-5,105))
    ax.set_yticks([-100,-50,0] if prefix=='low' else [0,50,100])
    ax.set_xticks(range(4));ax.set_xticklabels(classes,fontsize=8)
    ax.set_ylabel('Low-water impact (%)' if prefix=='low' else 'High-water impact (%)',fontsize=8)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    if prefix=='low':
        handles=[Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='#667278',label='Watershed'),
                 Line2D([],[],marker='D',ls='',color='#343B40',label='QTP'),
                 Line2D([],[],color='#657279',lw=1,label='Watershed median')]
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.14),
                  frameon=False,ncol=3,fontsize=6.8,borderaxespad=0)
        ax.set_title('Water-body size classes: low-water extremes',fontsize=8.5,loc='left',pad=8)
    else:
        ax.set_title('Water-body size classes: high-water extremes',fontsize=8.5,loc='left',pad=8)
        ax.text(0,-.34,'Filled: significant area trend; open: non-significant/undefined.\nSize classes are monthly regional aggregates.',
                transform=ax.transAxes,fontsize=6.4,va='top')
    add_panel_letter(ax,letter)


def draw_integrated_qtp_series(ax_area,ax_event,folder,region):
    """Use saved monthly flags and thresholds; no change to event definitions."""
    import matplotlib.dates as mdates
    df=pd.read_csv(folder/'Regional_monthly_components.csv')
    df=df[(df.Scale=='QTP')&(df.Region=='QTP')&(df.SizeClass=='Total')].copy()
    df['Date']=pd.to_datetime(df.Date,format='%Y-%m');df=df.sort_values('Date')
    if not pd.DatetimeIndex(df.Date).equals(EXPECTED_MONTHS):
        raise ValueError('Incomplete QTP total monthly series')
    for flag in ['LowEvent','HighEvent']:
        df[flag]=df[flag].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
        if df[flag].isna().any(): raise ValueError('Invalid event flag')
    r=region[(region.Scale=='QTP')&(region.Region=='QTP')&(region.SizeClass=='Total')].iloc[0]
    ax_area.plot(df.Date,df.Area_km2/1000,color='#9AA3A8',lw=.85,label='Monthly area')
    ax_area.plot(df.Date,df.Trend_km2/1000,color='#30383D',lw=1.5,label='STL trend')
    ax_event.plot(df.Date,df.AnnualDeviation_km2/1000,color='#9AA3A8',lw=.8)
    ax_event.plot(df.Date,df.LowThreshold_km2/1000,color=LOW_COLOUR,ls='--',lw=.8,label='P10')
    ax_event.plot(df.Date,df.HighThreshold_km2/1000,color=HIGH_COLOUR,ls='--',lw=.8,label='P90')
    for flag,colour,marker,label in [('LowEvent',LOW_COLOUR,'v','Low event'),('HighEvent',HIGH_COLOUR,'^','High event')]:
        sub=df[df[flag]]
        ax_area.scatter(sub.Date,sub.Area_km2/1000,s=15,c=colour,marker=marker,zorder=4,label=label)
        ax_event.scatter(sub.Date,sub.AnnualDeviation_km2/1000,s=15,c=colour,marker=marker,zorder=4)
    ax_area.set_ylabel('Area (10³ km²)',fontsize=8)
    ax_event.set_ylabel('Annual-mean deviation\n(10³ km²)',fontsize=8)
    ax_event.axhline(0,color='.65',lw=.5)
    ax_area.set_title(f'QTP total area: trend = {r.beta_yr:.1f} km² yr⁻¹',loc='left',fontsize=9,pad=7)
    ax_event.set_title(f'QTP extremes: I_low = {r.i_low:.1f}%; I_high = {r.i_high:.1f}%',loc='left',fontsize=9,pad=7)
    ax_area.legend(frameon=False,ncol=4,fontsize=6.1,loc='upper left',handlelength=1.3,columnspacing=.9)
    ax_event.legend(frameon=False,ncol=2,fontsize=6.7,loc='upper left')
    for ax,letter in [(ax_area,'c'),(ax_event,'d')]:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
        ax.margins(x=.015,y=.25)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=7)
        add_panel_letter(ax,letter)

def draw_region_size_matrix(ax,region,prefix,show_regions=True):
    """Merge former size and regional panels; cells are regional area-series indices.

    Cell numbers keep signed indices. Colour encodes |I| with the same 0–100 range
    for both tails. Asterisk denotes significant area trend (not significant I).
    """
    classes=['Total','Small','Medium','Large']
    order=['QTP']+sorted(set(region.Region)-{'QTP'})
    if region.duplicated(['Region','SizeClass']).any(): raise ValueError('Duplicate regional size rows')
    values=region.pivot(index='Region',columns='SizeClass',values='i_'+prefix).reindex(index=order,columns=classes)
    ps=region.pivot(index='Region',columns='SizeClass',values='mk_p').reindex(index=order,columns=classes)
    beta=region.pivot(index='Region',columns='SizeClass',values='beta_yr').reindex(index=order,columns=classes)
    cmap=LinearSegmentedColormap.from_list('impact_'+prefix,['#FAFAF8','#F1D4D8',LOW_COLOUR] if prefix=='low'
                                           else ['#FAFAF8','#C5E5EC',HIGH_COLOUR])
    cmap.set_bad('#E4E4E4')
    data=values.to_numpy(float)
    im=ax.imshow(np.ma.masked_invalid(np.abs(data)),cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for (i,j),value in np.ndenumerate(data):
        label='NA'
        if np.isfinite(value):
            sig='*' if ps.iloc[i,j]<EXTREME_ALPHA else ''
            direction='↑' if beta.iloc[i,j]>0 else '↓' if beta.iloc[i,j]<0 else '→'
            label=f'{value:.1f}{sig} {direction}'
        rgb=im.cmap(im.norm(abs(value)))[:3] if np.isfinite(value) else (1,1,1)
        colour='white' if np.dot(rgb,[.2126,.7152,.0722])<.53 else '#30383D'
        ax.text(j,i,label,ha='center',va='center',fontsize=7.2,color=colour)
    ax.set_xticks(range(4));ax.set_xticklabels(classes,fontsize=8)
    ax.set_yticks(range(len(order)));ax.set_yticklabels(order if show_regions else ['']*len(order),fontsize=8)
    ax.tick_params(length=0,pad=5)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title('Low-water impact (%)' if prefix=='low' else 'High-water impact (%)',fontsize=9,pad=8)
    bar=ax.figure.colorbar(im,ax=ax,fraction=.025,pad=.025,shrink=.85)
    bar.set_ticks([0,50,100]);bar.set_label('|Impact| (%)',fontsize=7);bar.ax.tick_params(labelsize=7)
    if show_regions:
        add_panel_letter(ax,'f')
        ax.text(0,-.13,'Numbers: signed impact (%).  ↑/↓: area trend.  *: Hamed–Rao MK p < 0.05.',
                transform=ax.transAxes,fontsize=7,va='top')


VIOLIN_TREND_EDGES = np.array([0.,5.,10.,25.,50.,np.inf])
VIOLIN_TREND_LABELS = ['0–5','5–10','10–25','25–50','≥50']
SAVE_DETAILED_STL = False  # Main figure emphasises seasonality; optional extra diagnostic.

def draw_grid_violin(ax,points,prefix,letter,horizontal=False):
    """Paired boxplots by |relative area trend| (name retained for compatibility).

    Only p<0.05 and direction-consistent events. Equal maximum violin width per
    group: boxes show Q1–Q3, median, 1.5-IQR whiskers and all outliers. For
    fewer than five or constant observations show points instead of a box.
    """
    from matplotlib.patches import Patch
    frame=points[(points.sig_chg!=0)&points[prefix+'_ok']&(points.mean_km2>0)].copy()
    frame['absrel']=abs(1000*frame.beta_yr/frame.mean_km2)
    frame=frame[np.isfinite(frame.absrel)&np.isfinite(frame['i_'+prefix])]
    frame['bin']=pd.cut(frame.absrel,VIOLIN_TREND_EDGES,right=False,labels=False)
    rows=[]
    for b,label in enumerate(VIOLIN_TREND_LABELS):
        for direction,offset,colour,name in [(-1,-.19,'#DA8D96','Decreasing'),(1,.19,'#87C39D','Increasing')]:
            v=frame.loc[(frame['bin']==b)&(frame.sig_chg==direction),'i_'+prefix].to_numpy(float)
            pos=b+offset;n=len(v)
            rows.append(dict(Event=prefix,Bin=label,Trend=name,Count=n,
                Median=np.median(v) if n else np.nan,Q25=np.quantile(v,.25) if n else np.nan,
                Q75=np.quantile(v,.75) if n else np.nan))
            if not n: continue
            if n>=5 and np.ptp(v)>1e-10:
                ax.boxplot([v],positions=[pos],widths=.30,patch_artist=True,manage_ticks=False,
                    whis=1.5,showfliers=True,vert=not horizontal,
                    boxprops=dict(facecolor=colour,edgecolor='#566167',linewidth=.7),
                    medianprops=dict(color='#29343A',linewidth=1.2),
                    whiskerprops=dict(color='#68757B',linewidth=.7),
                    capprops=dict(color='#68757B',linewidth=.7),
                    flierprops=dict(marker='.',markersize=1.8,markeredgecolor=colour,alpha=.45))
            else:
                jitter=pos+np.linspace(-.055,.055,n)
                ax.scatter(v if horizontal else jitter,jitter if horizontal else v,
                           s=5 if horizontal else 13,c=colour,edgecolors='#4E5558',linewidths=.3,zorder=4)
    if horizontal:
        ax.axvline(-50 if prefix=='low' else 50,color='.6',lw=.6,ls='--')
        ax.set_xlim((-103,3) if prefix=='low' else (-3,103))
        ax.set_xticks([-100,-50,0] if prefix=='low' else [0,50,100])
        ax.set_yticks(range(len(VIOLIN_TREND_LABELS)))
        ax.set_yticklabels(VIOLIN_TREND_LABELS)
        ax.set_ylim(len(VIOLIN_TREND_LABELS)-.4,-.6)
        ax.set_xlabel('Extreme impact (%)',fontsize=6,labelpad=2)
        ax.set_ylabel('Absolute area trend (% per decade)',fontsize=6,labelpad=2)
        ax.tick_params(labelsize=6,length=2,pad=2)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')
        if frame.empty:
            ax.text(.5,.5,'No eligible grids',transform=ax.transAxes,ha='center',fontsize=6)
        return rows
    ax.axhline(-50 if prefix=='low' else 50,color='.55',lw=.7,ls='--')
    ax.set_ylim((-103,3) if prefix=='low' else (-3,103))
    ax.set_yticks([-100,-75,-50,-25,0] if prefix=='low' else [0,25,50,75,100])
    ax.set_xlim(-.6,len(VIOLIN_TREND_LABELS)-.4)
    ax.set_xticks(range(len(VIOLIN_TREND_LABELS)));ax.set_xticklabels(VIOLIN_TREND_LABELS,fontsize=7)
    ax.set_xlabel('Absolute relative long-term change (% per decade)',fontsize=8)
    ax.set_ylabel('Low-water impact (%)' if prefix=='low' else 'High-water impact (%)',fontsize=8)
    ax.set_title('Low-water extremes: grid distributions' if prefix=='low' else 'High-water extremes: grid distributions',
                 loc='left',fontsize=9,pad=8)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.legend(handles=[Patch(facecolor='#DA8D96',edgecolor='#4E5558',label='Decreasing'),
                       Patch(facecolor='#87C39D',edgecolor='#4E5558',label='Increasing')],
              title='Significant area trend',title_fontsize=6.5,frameon=False,ncol=2,
              fontsize=6.5,loc='upper right',borderaxespad=.25)
    if frame.empty: ax.text(.5,.5,'No eligible grids',transform=ax.transAxes,ha='center',color='.5')
    add_panel_letter(ax,letter)
    return rows

def draw_seasonal_event_series(ax_season,ax_event,folder,region):
    import matplotlib.dates as mdates
    df=pd.read_csv(folder/'Regional_monthly_components.csv')
    df=df[(df.Scale=='QTP')&(df.Region=='QTP')&(df.SizeClass=='Total')].copy()
    df['Date']=pd.to_datetime(df.Date,format='%Y-%m');df=df.sort_values('Date')
    if not pd.DatetimeIndex(df.Date).equals(EXPECTED_MONTHS): raise ValueError('Incomplete QTP series')
    for flag in ['LowEvent','HighEvent']:
        df[flag]=df[flag].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
        if df[flag].isna().any(): raise ValueError('Invalid event flag')
    r=region[(region.Scale=='QTP')&(region.Region=='QTP')&(region.SizeClass=='Total')].iloc[0]
    ax_season.plot(df.Date,df.Seasonal_km2/1000,color=HIGH_COLOUR,lw=1)
    ax_season.fill_between(df.Date,df.Seasonal_km2/1000,0,color='#91CADA',alpha=.35)
    ax_season.set_ylabel('Seasonal component\n(10³ km²)',fontsize=8)
    ax_season.set_title('QTP total area: STL seasonal component',loc='left',fontsize=9,pad=8)
    ax_event.plot(df.Date,df.AnnualDeviation_km2/1000,color='#939CA2',lw=.8)
    ax_event.plot(df.Date,df.LowThreshold_km2/1000,color=LOW_COLOUR,ls='--',lw=.7,label='P10')
    ax_event.plot(df.Date,df.HighThreshold_km2/1000,color=HIGH_COLOUR,ls='--',lw=.7,label='P90')
    for flag,colour,marker,label in [('LowEvent',LOW_COLOUR,'v','Low event'),('HighEvent',HIGH_COLOUR,'^','High event')]:
        sub=df[df[flag]]
        ax_event.scatter(sub.Date,sub.AnnualDeviation_km2/1000,s=14,c=colour,marker=marker,zorder=4,label=label)
    ax_event.set_ylabel('Annual-mean deviation\n(10³ km²)',fontsize=8)
    ax_event.set_title(f'QTP extreme events: I_low = {r.i_low:.1f}%; I_high = {r.i_high:.1f}%',loc='left',fontsize=9,pad=8)
    ax_event.legend(frameon=False,ncol=4,fontsize=6,loc='upper left',columnspacing=.8,handlelength=1.3)
    for ax,letter in [(ax_season,'e'),(ax_event,'f')]:
        ax.axhline(0,color='.65',lw=.5);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);ax.margins(x=.015,y=.26)
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=7);add_panel_letter(ax,letter)

def draw_region_size_dots(ax,region):
    """Restore the previous regional point plot; marker shape adds size classes."""
    from matplotlib.lines import Line2D
    names=['QTP']+sorted(set(region.Region)-{'QTP'})
    classes=['Total','Small','Medium','Large'];markers=['o','^','s','D']
    for j,(size,marker) in enumerate(zip(classes,markers)):
        sub=region[region.SizeClass==size].set_index('Region').reindex(names)
        y=np.arange(len(names))+(j-1.5)*.17
        for prefix,colour in [('low',LOW_COLOUR),('high',HIGH_COLOUR)]:
            x=sub['i_'+prefix];valid=np.isfinite(x);sig=valid&(sub.mk_p<EXTREME_ALPHA)
            ax.scatter(x[valid],y[valid],marker=marker,s=22,facecolors='white',edgecolors=colour,linewidths=.8,zorder=3)
            ax.scatter(x[sig],y[sig],marker=marker,s=22,c=colour,edgecolors=colour,linewidths=.8,zorder=4)
    for x in [-50,50]:ax.axvline(x,color='.55',ls='--',lw=.7)
    ax.axvline(0,color='.75',lw=.6)
    ax.set_xlim(-103,103);ax.set_xticks([-100,-50,0,50,100])
    ax.set_yticks(range(len(names)));ax.set_yticklabels(names,fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('Extreme impact (%)',fontsize=8)
    ax.set_title('QTP and watersheds by water-body size',loc='left',fontsize=9,pad=9)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    shapes=[Line2D([],[],marker=m,ls='',markerfacecolor='white',markeredgecolor='#4E5558',label=s) for s,m in zip(classes,markers)]
    legend=ax.legend(handles=shapes,ncol=4,frameon=False,fontsize=6.8,loc='lower left',bbox_to_anchor=(0,1.07),borderaxespad=0)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([],[],color=LOW_COLOUR,lw=2,label='Low water'),Line2D([],[],color=HIGH_COLOUR,lw=2,label='High water')],
              ncol=2,frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.10))
    ax.text(.5,-.18,'Filled: significant area trend; open: non-significant/undefined.',
             transform=ax.transAxes,ha='center',fontsize=6.3)
    add_panel_letter(ax,'g')

def draw_combined_seasonality(ax,folder,region):
    import matplotlib.dates as mdates
    df=pd.read_csv(folder/'Regional_monthly_components.csv')
    df=df[(df.Scale=='QTP')&(df.Region=='QTP')&(df.SizeClass=='Total')].copy()
    df['Date']=pd.to_datetime(df.Date,format='%Y-%m');df=df.sort_values('Date')
    if not pd.DatetimeIndex(df.Date).equals(EXPECTED_MONTHS): raise ValueError('Incomplete QTP total series')
    ax.fill_between(df.Date,df.Seasonal_km2/1000,0,color='#91CADA',alpha=.25,zorder=1)
    ax.plot(df.Date,df.Seasonal_km2/1000,color='#559CAF',lw=.85,label='STL seasonal component',zorder=2)
    ax.plot(df.Date,df.AnnualDeviation_km2/1000,color='#687279',lw=.85,label='Annual-mean deviation',zorder=3)
    for flag,colour,marker,label in [('LowEvent',LOW_COLOUR,'v','Low event'),('HighEvent',HIGH_COLOUR,'^','High event')]:
        flags=df[flag].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
        if flags.isna().any(): raise ValueError('Invalid event flag')
        sub=df[flags]
        ax.scatter(sub.Date,sub.AnnualDeviation_km2/1000,s=24,c=colour,marker=marker,zorder=5,label=label)
    r=region[(region.Scale=='QTP')&(region.Region=='QTP')&(region.SizeClass=='Total')].iloc[0]
    for prefix,colour,offset in [('low',LOW_COLOUR,(7,-17)),('high',HIGH_COLOUR,(7,10))]:
        date=pd.to_datetime(r[prefix+'_date']);match=df[df.Date==date]
        if len(match)==1:
            ax.annotate(date.strftime('%Y-%m'),(date,match.iloc[0].AnnualDeviation_km2/1000),
                        xytext=offset,textcoords='offset points',fontsize=7,color=colour,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    ax.axhline(0,color='.7',lw=.5)
    ax.set_ylabel('Area deviation (10³ km²)',fontsize=8)
    ax.set_title('QTP seasonality and extreme events',loc='left',fontsize=9,pad=8)
    ax.legend(frameon=False,ncol=2,fontsize=6.7,loc='upper left',handlelength=1.5,columnspacing=1)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);ax.margins(x=.015,y=.36)
    ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=7);add_panel_letter(ax,'e')

def draw_qtp_size_only(ax,region):
    """Four QTP size aggregates only; basin information remains in the output CSV."""
    classes=['Total','Small','Medium','Large']
    sub=region[(region.Scale=='QTP')&(region.Region=='QTP')].set_index('SizeClass').reindex(classes)
    for prefix,colour,offset in [('low',LOW_COLOUR,-.10),('high',HIGH_COLOUR,.10)]:
        values=sub['i_'+prefix].to_numpy(float);y=np.arange(4)+offset
        good=np.isfinite(values)
        ax.hlines(y[good],0,values[good],color=colour,alpha=.35,lw=2)
        ax.scatter(values[good],y[good],s=36,c=colour,edgecolors='white',linewidths=.6,zorder=4,
                   label='Low water' if prefix=='low' else 'High water')
        for value,yy in zip(values[good],y[good]):
            ax.annotate(f'{value:.1f}',(value,yy),xytext=(-5 if prefix=='low' else 5,0),
                        textcoords='offset points',ha='right' if prefix=='low' else 'left',va='center',fontsize=7)
    for x in [-50,50]: ax.axvline(x,color='.6',ls='--',lw=.7)
    ax.axvline(0,color='.75',lw=.5)
    ax.set_xlim(-100,100);ax.set_xticks([-100,-50,0,50,100])
    ax.set_yticks(range(4));ax.set_yticklabels(classes);ax.set_ylim(3.5,-.65)
    ax.set_xlabel('Extreme impact (%)',fontsize=8)
    ax.set_title('QTP water-body size classes',loc='left',fontsize=9,pad=8)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.20))
    ax.tick_params(labelsize=7);add_panel_letter(ax,'f')


def clean_combined_axis(ax,letter,remove_notes=False):
    ax.set_title('');ax.set_title('',loc='left');ax.set_title('',loc='right')
    for text in list(ax.texts):
        value=text.get_text()
        if value in list('abcdefghij') or (remove_notes and
            (value.startswith('Colour:') or value.startswith('Filled:') or value.startswith('Numbers:'))):
            text.remove()
    ax.text(-.045,1.015,letter,transform=ax.transAxes,fontsize=12,fontweight='bold',
            ha='right',va='bottom',zorder=100)


def run_combined_figure(explicit=None):
    from matplotlib.patches import Patch
    seasonality,change,qtp=read_data()
    qtp_cseason=get_qtp_cseason()
    inc_pct,dec_pct=calculate_change_proportions(change)
    seasonal_sig=get_significant_cells(change)
    reg=pd.read_csv(REGIONAL_CSV,encoding='utf-8-sig')
    required={'Region','SizeClass','Cseason_pct','RelDec_pct_per_decade'}
    if not required.issubset(reg):raise ValueError(f'Missing regional metrics: {required-set(reg)}')
    reg['Region']=reg.Region.astype(str).str.strip()
    reg['SizeClass']=reg.SizeClass.astype(str).str.strip().str.capitalize()
    contributions=read_size_contributions()
    folder=find_extreme_directory(explicit)
    grid,points,region=compute_significance(folder)
    if points.crs!=seasonality.crs:
        raise ValueError('Seasonality and extreme points must use the same CRS; no statistical layer is reprojected')
    mapped=points.loc[(points.status=='OK')&(points.map_ok==1)&(points.n_valid==120)].copy()
    if mapped.empty: raise ValueError('No eligible grid cells')
    counts=significant_impact_counts(mapped)
    out=Path(r'D:\phdl2\QTP surface water dynamics\code availability\test_output');out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(12.8,10.8),facecolor='white')
    # Fixed map rectangles give identical geographic scale across a, b, e, f.

    map_x=[.055,.535]
    map_width=.319
    box_offset=.356
    box_width=.080
    column_width=box_offset+box_width
    axes=[fig.add_axes([x,.785,map_width,.208]) for x in map_x]
    axes += [fig.add_axes([map_x[0]+.035,.554,.370,.164]),
             fig.add_axes([map_x[1]+.020,.554,.365,.164])]
    axes += [fig.add_axes([x,.303,map_width,.208]) for x in map_x]
    axes += [fig.add_axes([map_x[0]+.030,.05,.540,.191]),
             fig.add_axes([.705,.05,.265,.191])]
    extent=get_extent(qtp)
    draw_panel_a(axes[0],seasonality,qtp,extent,qtp_cseason)
    draw_panel_b(axes[1],change,seasonal_sig,qtp,extent,inc_pct,dec_pct)
    draw_regional_summary(axes[2],reg)
    axes[2].set_xlabel('Seasonality contribution (%)',fontsize=8)
    axes[2].set_ylabel('Relative change in seasonality\n(% per decade)',fontsize=8)
    draw_contribution_option(axes[3],contributions,fig,'heatmap_transposed')
    extreme_map(axes[4],mapped,qtp,extent,'low','e')
    extreme_map(axes[5],mapped,qtp,extent,'high','f')

    for idx,y in [(0,.785),(1,.785),(4,.303),(5,.303)]:
        axes[idx].set_position([map_x[idx % 2],y,map_width,.208])
        axes[idx].set_anchor('C')

    extra_axes=[ax for ax in fig.axes if ax not in axes]
    for cbax,rect in zip(extra_axes,[[map_x[0]+map_width+.018,.80,.006,.164],
                                  [map_x[1]+map_width+.018,.80,.006,.164],
                                  [map_x[1]+.398,.554,.006,.164]]):
        cbax.set_position(rect)
    axes[3].set_position([map_x[1]+.020,.554,.365,.164])
    box_axes=[];rows=[];map_keys=[]
    for i,(ax,prefix) in enumerate(zip(axes[4:6],['low','high'])):

        for child in list(ax.child_axes):
            child.remove()
        key_parent=fig.add_axes([map_x[i]+.027,.290,.045,.052],frameon=False)
        key_parent.set_axis_off()
        four_square_legend(key_parent,prefix)
        key=key_parent.child_axes[-1]
        key.set_axes_locator(None)
        key.set_position([map_x[i]+.027,.290,.045,.052])
        map_keys.append(key)
        box=fig.add_axes([map_x[i]+box_offset,.323,box_width,.164])
        rows+=draw_grid_violin(box,mapped,prefix,'',horizontal=True)
        box.legend(handles=[Patch(facecolor='#DA8D96',edgecolor='#566167',label='Decreasing'),
                            Patch(facecolor='#87C39D',edgecolor='#566167',label='Increasing')],
            loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,
            fontsize=5.5,handlelength=1.1,columnspacing=.7,borderaxespad=0)
        box_axes.append(box)
    draw_combined_seasonality(axes[6],folder,region)
    draw_qtp_size_only(axes[7],region)
    axes[7].set_ylim(3.5,-1.0)
    axes[7].legend(frameon=False,ncol=2,fontsize=7,loc='upper right',borderaxespad=.4)
    for ax,letter in zip(axes,'abcdefgh'):
        clean_combined_axis(ax,letter,remove_notes=True)

        for text in list(ax.texts):
            if text.get_text()==letter: text.remove()

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    inverse=fig.transFigure.inverted()
    letter_x=[]
    for key in map_keys:
        bounds=key.yaxis.label.get_window_extent(renderer)
        letter_x.append(inverse.transform(((bounds.x0+bounds.x1)/2,bounds.y1))[0])
    for row,y in enumerate([.986,.721,.514]):
        for col,x in enumerate(letter_x):
            fig.text(x,y,'abcdef'[2*row+col],fontsize=12,fontweight='bold',
                     ha='center',va='bottom')

    for ax,letter in zip(axes[6:8],'gh'):
        bounds=ax.get_tightbbox(renderer)
        x=inverse.transform((bounds.x0,bounds.y0))[0]+.005
        fig.text(x,.244,letter,fontsize=12,fontweight='bold',ha='left',va='bottom')

    for extension in ['png','pdf','svg']:
        fig.savefig(out/('Figure4.'+extension),dpi=EXTREME_DPI,facecolor='white',
                    bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    print(f'[SAVE] {out / "Figure4.png"}')


if __name__ == '__main__':
    run_combined_figure(EXTREME_DIR)

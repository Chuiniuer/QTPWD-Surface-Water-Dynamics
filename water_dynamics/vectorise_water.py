# -*- coding: utf-8 -*-
"""
QTP 10m Albers water mask -> waterbody polygon
ArcGIS Pro Notebook stable version

2023-2025 only
"""

import arcpy
import re
import time
import gc
import csv
from pathlib import Path
from arcpy.sa import SetNull



INPUT_ROOT = Path(
    r"E:\phd_l3\QTPWD_alberts_equal_area"
)

OUTPUT_ROOT = Path(
    r"E:\phd_l3\QTP_waterbody_vector_2023_2025"
)

SCRATCH_DIR = OUTPUT_ROOT / "_scratch"
LOG_DIR = OUTPUT_ROOT / "_logs"

SUMMARY_CSV = OUTPUT_ROOT / "waterbody_summary_2023_2025.csv"


TARGET_YEARS = {
    2023,
    2024,
    2025
}


AREA_FIELD = "Area_km2"



OUTPUT_ROOT.mkdir(exist_ok=True)
SCRATCH_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


arcpy.env.overwriteOutput = True
arcpy.env.scratchWorkspace = str(SCRATCH_DIR)


try:
    arcpy.env.addOutputsToMap = False
except:
    pass


try:
    arcpy.CheckOutExtension("Spatial")
except:
    pass




def find_inputs():

    files = []

    for tif in INPUT_ROOT.rglob("*.tif"):

        m = re.search(
            r"(20\d{2})[_-](0[1-9]|1[0-2])",
            tif.name
        )

        if m:

            year = int(m.group(1))
            month = int(m.group(2))

            if year in TARGET_YEARS:
                files.append(
                    (year, month, tif)
                )

    files.sort(
        key=lambda x:(x[0],x[1])
    )

    return files



tasks = find_inputs()


print("="*80)
print("QTP waterbody vectorization")
print("="*80)

print("Input:")
print(INPUT_ROOT)

print("Output:")
print(OUTPUT_ROOT)

print("Years:")
print(sorted(TARGET_YEARS))

print("Files:")
print(len(tasks))

print("="*80)



if len(tasks)!=36:

    print(
        "Warning:",
        "expected 36 files, found",
        len(tasks)
    )




def get_gdb(year):

    year_dir = OUTPUT_ROOT / str(year)

    year_dir.mkdir(
        exist_ok=True
    )

    gdb = year_dir / (
        f"QTP_waterbody_{year}.gdb"
    )


    if not arcpy.Exists(str(gdb)):

        arcpy.management.CreateFileGDB(
            str(year_dir),
            gdb.name
        )


    return gdb



def get_fc(year,month):

    return (
        get_gdb(year)
        /
        f"WB_{year}_{month:02d}"
    )



def get_done(year,month):

    return (
        LOG_DIR
        /
        f"{year}_{month:02d}.done"
    )



def write_summary(row):

    exists = SUMMARY_CSV.exists()

    with open(
        SUMMARY_CSV,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        w=csv.writer(f)

        if not exists:

            w.writerow(
                [
                    "Year",
                    "Month",
                    "Status",
                    "Waterbody_Count",
                    "Time_min"
                ]
            )


        w.writerow(row)






def process_month(year,month,tif):


    start=time.time()


    out_fc=get_fc(
        year,
        month
    )


    done=get_done(
        year,
        month
    )


    if (
        done.exists()
        and
        arcpy.Exists(str(out_fc))
    ):

        print(
            "SKIP",
            year,
            month
        )

        return "SKIP"



    print("\nSTART",
          year,
          month)


    r=None
    water=None


    try:


        # VALUE=0/2 -> NoData

        r = arcpy.sa.Raster(
            str(tif)
        )


        water = SetNull(
            r != 1,
            1
        )



        # Raster -> Polygon

        arcpy.conversion.RasterToPolygon(
            water,
            str(out_fc),
            "SIMPLIFY",
            "VALUE",
            "SINGLE_OUTER_PART"
        )



        del water
        water=None

        del r
        r=None

        gc.collect()




        fields = {
            f.name
            for f in arcpy.ListFields(
                str(out_fc)
            )
        }


        if AREA_FIELD not in fields:

            arcpy.management.AddField(
                str(out_fc),
                AREA_FIELD,
                "DOUBLE"
            )



        arcpy.management.CalculateField(
            str(out_fc),
            AREA_FIELD,
            "!shape.area@SQUAREKILOMETERS!",
            "PYTHON3"
        )



        count=int(
            arcpy.management.GetCount(
                str(out_fc)
            )[0]
        )



        t=(time.time()-start)/60



        done.write_text(
            f"DONE\ncount={count}\ntime={t}",
            encoding="utf-8"
        )



        write_summary(
            [
                year,
                month,
                "DONE",
                count,
                round(t,3)
            ]
        )



        print(
            "DONE",
            year,
            month,
            "features:",
            count,
            "time:",
            round(t,2),
            "min"
        )


        return "DONE"



    except Exception as e:


        print(
            "ERROR",
            year,
            month
        )

        print(e)

        try:
            print(
                arcpy.GetMessages(2)
            )
        except:
            pass


        write_summary(
            [
                year,
                month,
                "ERROR",
                "",
                ""
            ]
        )


        return "ERROR"



    finally:


        water=None
        r=None

        gc.collect()


        try:
            arcpy.management.ClearWorkspaceCache()
        except:
            pass


        time.sleep(2)






done=0
skip=0
error=0


total_start=time.time()



for i,(year,month,tif) in enumerate(tasks,1):


    print(
        "\n",
        "="*70
    )

    print(
        f"[{i}/{len(tasks)}]",
        year,
        month
    )


    result=process_month(
        year,
        month,
        tif
    )


    if result=="DONE":
        done+=1

    elif result=="SKIP":
        skip+=1

    else:
        error+=1




hours=(
    time.time()-total_start
)/3600



print("\n")
print("="*80)
print("FINISHED")
print("="*80)

print(
    "DONE:",
    done
)

print(
    "SKIP:",
    skip
)

print(
    "ERROR:",
    error
)

print(
    "Time:",
    round(hours,2),
    "hours"
)

print(
    "Summary:",
    SUMMARY_CSV
)

print("="*80)


try:
    arcpy.CheckInExtension("Spatial")
except:
    pass
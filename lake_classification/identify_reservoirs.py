# -*- coding: utf-8 -*-
"""CRD reference-based monthly reservoir separation. Python 3.9+.
Dependencies: fiona, shapely>=2, pyproj. Run directly; original inputs unchanged.
"""
import argparse
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import sys
import traceback
import subprocess
import time
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

CRD = Path(r'D:\phdl2\QTP surface water dynamics\水库数据\CRD v1.1')
WATER = Path(r'E:\phd_l3\QTPWD_vector')
BOUNDARY = Path(r'D:\phdl2\QTP surface water dynamics\regions\QTP.shp')
OUT = Path(r'E:\phd_l3\QTP_reservoir_separation')
VERSION = '1.1'


def log(s):
    print(s, flush=True)


def signature(path):
    paths = sorted(p for p in path.rglob('*') if p.is_file() and not p.name.endswith('.lock')) if path.is_dir() else sorted(p for p in path.parent.glob(path.stem + '.*') if p.suffix.lower() in ('.shp', '.shx', '.dbf', '.prj', '.cpg'))
    return [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stamp(p):
    s = p.stat()
    return [s.st_size, s.st_mtime_ns]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crd-dir', type=Path, default=CRD)
    ap.add_argument('--crd-shp', type=Path)
    ap.add_argument('--crd-id-field', help='Optional exact CRD identifier attribute name')
    ap.add_argument('--water-dir', type=Path, default=WATER)
    ap.add_argument('--boundary', type=Path, default=BOUNDARY)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--threshold', type=float, default=0.5, help='Reservoir if union overlap / QTP-clipped object area > threshold')
    ap.add_argument('--years', nargs='+', type=int, default=list(range(2016, 2026)))
    ap.add_argument('--workers', type=int, default=2, help='Independent monthly processes; use 1 on a slow disk')
    ap.add_argument('--batch-size', type=int, default=2000)
    ap.add_argument('--only-layer', help=argparse.SUPPRESS)
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.workers <= 8 or not 100 <= args.batch_size <= 20000:
        raise ValueError('workers: 1..8; batch-size: 100..20000')
    if not 0 < args.threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    import fiona
    import numpy as np
    import shapely
    from shapely.geometry import shape, mapping, MultiPolygon
    from shapely.ops import transform, unary_union
    from shapely import make_valid
    from shapely.strtree import STRtree
    from shapely.prepared import prep
    from pyproj import CRS, Transformer
    if int(shapely.__version__.split('.')[0]) < 2:
        raise RuntimeError('Requires shapely >= 2: python -m pip install --upgrade shapely')
    for p in (args.crd_dir, args.water_dir, args.boundary):
        if not p.exists():
            raise FileNotFoundError(p)
    crds = [args.crd_shp] if args.crd_shp else sorted(p for p in args.crd_dir.rglob('*') if p.suffix.lower() == '.shp' and 'all_reservoir' in p.stem.lower())
    if len(crds) != 1:
        raise RuntimeError(f'Expected one all_reservoirs SHP, found {crds}. Set --crd-shp explicitly.')
    crd = crds[0]
    args.output.mkdir(parents=True, exist_ok=True)

    # Discover and verify monthly layers before starting expensive processing.
    jobs = []
    target = None
    for year in args.years:
        gdb = args.water_dir / str(year) / f'QTP_waterbody_{year}.gdb'
        if not gdb.exists():
            raise FileNotFoundError(gdb)
        layers = []
        for name in fiona.listlayers(str(gdb)):
            with fiona.open(str(gdb), layer=name) as src:
                if 'Polygon' not in src.schema['geometry']:
                    continue
                if not src.crs_wkt:
                    raise ValueError(f'Missing CRS: {gdb}, {name}')
                c = CRS.from_wkt(src.crs_wkt)
                if target is None:
                    target = c
                if not target.equals(c, ignore_axis_order=True):
                    raise ValueError(f'Inconsistent monthly CRS: {name}')
                layers.append(name)
        if len(layers) != 12:
            raise ValueError(f'{gdb}: expected 12 monthly polygon layers; found {len(layers)}: {layers}')
        # Preserve source layer names instead of guessing date strings.
        layers.sort()
        for name in layers:
            jobs.append((year, gdb, name))
        log(f'[DISCOVER] {year}: {layers}')
    if target is None or not target.is_projected or 'albers' not in (target.coordinate_operation.method_name.lower() if target.coordinate_operation else ''):
        raise ValueError('Monthly reference must have QTP Albers projected CRS')
    if any(abs(ax.unit_conversion_factor - 1) > 1e-9 for ax in target.axis_info[:2]):
        raise ValueError('Projected units must be metres')
    log(f'[CRS] {target.name}')

    def polygon(g):
        if not g.is_valid:
            g = make_valid(g)
        if g.geom_type == 'Polygon':
            return MultiPolygon([g]) if g.area else MultiPolygon([])
        if g.geom_type == 'MultiPolygon':
            return g
        if g.geom_type == 'GeometryCollection':
            return MultiPolygon([p for sub in g.geoms for p in polygon(sub).geoms])
        return MultiPolygon([])

    def projected(g, tf):
        return polygon(transform(tf.transform, shape(g)))

    with fiona.open(str(args.boundary)) as src:
        if not src.crs_wkt:
            raise ValueError('QTP boundary has no CRS')
        tf = Transformer.from_crs(src.crs_wkt, target, always_xy=True)
        boundary = unary_union([projected(ft['geometry'], tf) for ft in src if ft['geometry']])
    if boundary.is_empty:
        raise ValueError('QTP boundary is empty')
    prepared = prep(boundary)
    reference_key = digest([VERSION, signature(crd), signature(args.boundary), target.to_wkt(), args.crd_id_field])
    reference = args.output / 'CRD_QTP_reference.gpkg'
    ref_meta = args.output / 'CRD_QTP_reference.json'
    reuse = False
    if reference.exists() and ref_meta.exists() and not args.rebuild:
        m = json.loads(ref_meta.read_text('utf-8'))
        reuse = m.get('key') == reference_key and m.get('stamp') == stamp(reference)
    if not reuse:
        tmp = reference.with_name(reference.stem + '.working.gpkg')
        tmp.unlink(missing_ok=True)
        with fiona.open(str(crd)) as src:
            if not src.crs_wkt:
                raise ValueError('CRD has no CRS')
            if 'Polygon' not in src.schema['geometry']:
                raise ValueError('CRD input must be reservoir polygons, not points')
            fields = list(src.schema['properties'])
            log(f'[CRD FIELDS] {fields}')
            normalized = {re.sub(r'[^a-z0-9]', '', n.lower()): n for n in fields}
            if args.crd_id_field:
                if args.crd_id_field not in fields:
                    raise ValueError(f'Configured ID field not found: {args.crd_id_field}; available: {fields}')
                source_id = args.crd_id_field
            else:
                source_id = next((normalized[n] for n in ('id', 'crdid', 'resid', 'resvid', 'reservoirid', 'objectid', 'fid') if n in normalized), None)
            id_field = 'q_ref_id'
            while id_field.lower() in {n.lower() for n in fields}:
                id_field += '_'
            log(f'[CRD ID] {source_id if source_id else "source feature FID (not official CRD ID)"}; saved as {id_field}')
            seen_ids = set()
            tf = Transformer.from_crs(src.crs_wkt, target, always_xy=True)
            schema = dict(src.schema)
            schema['geometry'] = 'MultiPolygon'
            schema['properties'] = dict(src.schema['properties'])
            schema['properties'][id_field] = 'str'
            with fiona.open(str(tmp), 'w', driver='GPKG', layer='reservoir_reference', schema=schema, crs_wkt=target.to_wkt(), FID='__ogr_pk') as dst:
                n = 0
                for ft in src:
                    if not ft['geometry']:
                        continue
                    g = projected(ft['geometry'], tf)
                    if not g.is_empty and prepared.intersects(g):
                        g = polygon(g.intersection(boundary))
                        if not g.is_empty:
                            props = dict(ft['properties'])
                            value = props.get(source_id) if source_id else None
                            ref_id = str(value).strip() if value is not None else ''
                            if not ref_id or ref_id in seen_ids:
                                ref_id = 'source_fid:' + str(ft.id)
                            while ref_id in seen_ids:
                                ref_id += '_'
                            seen_ids.add(ref_id)
                            props[id_field] = ref_id
                            dst.write({'geometry': mapping(g), 'properties': props})
                            n += 1
                log(f'[CRD] {n} reservoirs intersect QTP')
        os.replace(tmp, reference)
        ref_meta.write_text(json.dumps({'key': reference_key, 'stamp': stamp(reference), 'id_field': id_field}), 'utf-8')
    else:
        log('[CACHE] CRD QTP reference reused')
        id_field = json.loads(ref_meta.read_text('utf-8'))['id_field']
    geoms, ids = [], []
    with fiona.open(str(reference), layer='reservoir_reference') as src:
        for ft in src:
            geoms.append(shape(ft['geometry']))
            ids.append(str(ft['properties'][id_field]))
    if not geoms:
        raise ValueError('No CRD polygons intersect QTP')
    tree = STRtree(geoms)
    gdb_keys = {str(gdb): digest(signature(gdb)) for gdb in sorted({g for _, g, _ in jobs})}
    if args.only_layer:
        jobs = [j for j in jobs if j[2] == args.only_layer]
        if len(jobs) != 1:
            raise ValueError('Worker layer must identify exactly one monthly layer')
    if args.workers > 1 and not args.only_layer:
        # The parent has already prepared the reference. Children only read it.
        # Threads here supervise separate Python processes; GIS work is multiprocess.
        (args.output / 'README_fast.txt').write_text(
            'Two workers by default; use --workers 1 for serial processing.\n'
            'Reservoir candidates exceed the overlap threshold; other objects may include rivers and missed reservoirs.\n'
            'Use q_area for clipped area and q_ovfrac for overlap.\n', encoding='utf-8')
        logdir = args.output / 'logs'
        logdir.mkdir(exist_ok=True)
        def run_month(job):
            year, _, name = job
            safe = re.sub(r'[^\w.-]', '_', name)
            report = args.output / 'reports' / str(year) / (safe + '.json')
            if report.exists() and not args.rebuild:
                old = json.loads(report.read_text('utf-8'))
                key = digest([reference_key, gdb_keys[str(job[1])], name, args.threshold])
                outputs = old.get('outputs', {})
                if old.get('key') == key and len(outputs) == 3 and all(Path(p).exists() and stamp(Path(p)) == st for p, st in outputs.items()):
                    log(f'[CACHE] {year}/{name}')
                    return old['summary']
            logfile = logdir / f'{year}_{safe}.log'
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--workers', '1', '--years', str(year), '--only-layer', name,
                   '--crd-dir', str(args.crd_dir), '--crd-shp', str(crd), '--water-dir', str(args.water_dir),
                   '--boundary', str(args.boundary), '--output', str(args.output), '--threshold', str(args.threshold),
                   '--batch-size', str(args.batch_size)]
            if args.crd_id_field:
                cmd += ['--crd-id-field', args.crd_id_field]
            # Force a month rebuild by removing only its completion manifest, not reference.
            if args.rebuild:
                report.unlink(missing_ok=True)
            started = time.perf_counter()
            log(f'[START MONTH] {year}/{name}; log={logfile}')
            env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with logfile.open('w', encoding='utf-8') as f:
                child = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                while True:
                    try:
                        rc = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        log(f'[RUNNING] {name}: {(time.perf_counter()-started)/60:.1f} min; details in {logfile.name}')
            if rc:
                raise RuntimeError(f'{name} failed (exit {rc}); inspect {logfile}')
            result = json.loads(report.read_text('utf-8'))['summary']
            log(f'[DONE MONTH] {name}: {(time.perf_counter()-started)/60:.1f} min')
            return result
        results, failures = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_month, j): j for j in jobs}
            for f in as_completed(pending):
                try:
                    results.append(f.result())
                except Exception as exc:
                    failures.append(str(exc))
                    log(f'[FAILED] {exc}')
        if results:
            keys = list(dict.fromkeys(k for item in results for k in item))
            with (args.output / 'monthly_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(sorted(results, key=lambda x: (x['year'], x['layer'])))
        (args.output / 'parallel_run_report.json').write_text(json.dumps({'completed': len(results), 'failed': failures}, indent=2), 'utf-8')
        if failures:
            raise RuntimeError('Some months failed; completed months are cached. See logs and parallel_run_report.json')
        log(f'[SUCCESS] All {len(results)} months completed: {args.output}')
        return
    summaries = []
    extra = {'q_srcfid': 'str', 'q_layer': 'str', 'q_year': 'int', 'q_area': 'float', 'q_ovarea': 'float', 'q_ovfrac': 'float', 'q_crdids': 'str', 'q_ncrd': 'int', 'q_status': 'str'}
    for year, gdb, name in jobs:
        safe = re.sub(r'[^\w.-]', '_', name)
        res = args.output / 'reservoirs' / str(year) / (safe + '.gpkg')
        other = args.output / 'water_without_reservoirs' / str(year) / (safe + '.gpkg')
        report = args.output / 'reports' / str(year) / (safe + '.json')
        pairs = report.with_suffix('.csv')
        for p in (res, other, report):
            p.parent.mkdir(parents=True, exist_ok=True)
        key = digest([reference_key, gdb_keys[str(gdb)], name, args.threshold])
        if report.exists() and not args.rebuild:
            old = json.loads(report.read_text('utf-8'))
            if old.get('key') == key and all(p.exists() and old.get('outputs', {}).get(str(p)) == stamp(p) for p in (res, other, pairs)):
                summaries.append(old['summary'])
                log(f'[CACHE] {year}/{name}')
                continue
        log(f'[PROCESS] {year}/{name}')
        temps = [p.with_name(p.stem + '.working' + p.suffix) for p in (res, other, pairs)]
        for p in temps:
            p.unlink(missing_ok=True)
        summary = dict(year=year, layer=name, read=0, outside_or_empty=0, repaired=0, reservoirs=0, others=0, ambiguous_others=0, reservoir_km2=0., other_km2=0.)
        with ExitStack() as stack:
            src = stack.enter_context(fiona.open(str(gdb), layer=name))
            collisions = set(n.lower() for n in src.schema['properties']) & set(extra)
            if collisions:
                raise ValueError(f'Output attribute collision: {collisions}')
            schema = {'geometry': 'MultiPolygon', 'properties': {**src.schema['properties'], **extra}}
            writers = [stack.enter_context(fiona.open(str(p), 'w', driver='GPKG', layer=layer, schema=schema, crs_wkt=target.to_wkt(), FID='__ogr_pk')) for p, layer in zip(temps[:2], ('reservoirs', 'water_without_reservoirs'))]
            csvfile = stack.enter_context(temps[2].open('w', encoding='utf-8-sig', newline=''))
            cw = csv.writer(csvfile)
            cw.writerow(['source_fid', 'CRD_reference_ID', 'intersection_km2', 'fraction_of_water_object', 'decision'])
            batches = [[], []]
            timing = {'read_geometry_s': 0., 'boundary_s': 0., 'match_s': 0., 'write_s': 0.}
            started = last_progress = time.perf_counter()
            def objects():
                iterator = iter(src)
                while True:
                    t = time.perf_counter()
                    features = list(islice(iterator, args.batch_size))
                    if not features:
                        return
                    arr = np.empty(len(features), dtype=object)
                    arr[:] = [shape(ft['geometry']) if ft['geometry'] else None for ft in features]
                    valid = shapely.is_valid(arr)
                    for i in np.flatnonzero(~valid):
                        if arr[i] is not None:
                            arr[i] = make_valid(arr[i])
                            summary['repaired'] += 1
                    timing['read_geometry_s'] += time.perf_counter() - t
                    t = time.perf_counter()
                    hits = shapely.intersects(boundary, arr)
                    covered = shapely.covers(boundary, arr)
                    timing['boundary_s'] += time.perf_counter() - t
                    for ft, raw, hit, inside in zip(features, arr, hits, covered):
                        summary['read'] += 1
                        if raw is None or not hit:
                            summary['outside_or_empty'] += 1
                            continue
                        t = time.perf_counter()
                        g = polygon(raw if inside else raw.intersection(boundary))
                        timing['boundary_s'] += time.perf_counter() - t
                        if g.is_empty or g.area <= 0:
                            summary['outside_or_empty'] += 1
                            continue
                        yield ft, g
            shapely.prepare(boundary)
            for ft, g in objects():
                match_start = time.perf_counter()
                overlaps, matches = [], []
                for j in tree.query(g, predicate='intersects'):
                    cut = g.intersection(geoms[int(j)])
                    if cut.area > 0:
                        overlaps.append(cut)
                        matches.append((ids[int(j)], cut.area))
                ov = (overlaps[0].area if len(overlaps) == 1 else unary_union(overlaps).area) if overlaps else 0.
                timing['match_s'] += time.perf_counter() - match_start
                frac = min(1., max(0., ov / g.area))
                is_res = frac > args.threshold
                status = 'CRD_reservoir_candidate' if is_res else ('partial_overlap_review' if matches else 'no_CRD_overlap')
                ix = 0 if is_res else 1
                summary['reservoirs' if is_res else 'others'] += 1
                summary['reservoir_km2' if is_res else 'other_km2'] += g.area / 1e6
                summary['ambiguous_others'] += int(bool(matches) and not is_res)
                props = dict(ft['properties'])
                props.update(q_srcfid=str(ft.id), q_layer=name, q_year=year, q_area=g.area/1e6, q_ovarea=ov/1e6, q_ovfrac=frac, q_crdids=';'.join(sorted(set(i for i, _ in matches))), q_ncrd=len(matches), q_status=status)
                batches[ix].append({'geometry': mapping(g), 'properties': props})
                for rid, area in matches:
                    cw.writerow([ft.id, rid, area/1e6, area/g.area, status])
                if len(batches[ix]) >= args.batch_size:
                    t = time.perf_counter()
                    writers[ix].writerecords(batches[ix])
                    timing['write_s'] += time.perf_counter() - t
                    batches[ix].clear()
                if time.perf_counter() - last_progress >= 15:
                    last_progress = time.perf_counter()
                    log(f"  speed={summary['read']/max(1., last_progress-started):,.0f} objects/s; elapsed={(last_progress-started)/60:.1f} min")
                    log(f"  read={summary['read']:,}; reservoirs={summary['reservoirs']:,}; other={summary['others']:,}")
            for w, batch in zip(writers, batches):
                if batch:
                    t = time.perf_counter()
                    w.writerecords(batch)
                    timing['write_s'] += time.perf_counter() - t
        log(f'[TIMING] {timing}; total={time.perf_counter()-started:.1f}s')
        for p, count in zip(temps[:2], (summary['reservoirs'], summary['others'])):
            with fiona.open(str(p)) as check:
                if len(check) != count:
                    raise RuntimeError('Output feature count check failed')
        if summary['read'] != summary['outside_or_empty'] + summary['reservoirs'] + summary['others']:
            raise RuntimeError('Partition count check failed')
        for tmp, final in zip(temps, (res, other, pairs)):
            os.replace(tmp, final)
        report.write_text(json.dumps({'key': key, 'threshold': args.threshold, 'summary': summary, 'timing': timing, 'outputs': {str(p): stamp(p) for p in (res, other, pairs)}}, indent=2), 'utf-8')
        summaries.append(summary)
        log(f"[DONE] reservoirs={summary['reservoirs']:,}; other={summary['others']:,}; partial overlap={summary['ambiguous_others']:,}")
    if args.only_layer:
        log('[WORKER SUCCESS] Monthly outputs committed')
        return
    with (args.output / 'monthly_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    (args.output / 'README.txt').write_text(
        'Monthly reservoir candidates are identified from the CRD reference.\n'
        f'An object is assigned as a reservoir when overlap exceeds {args.threshold} of its area.\n'
        'Partially overlapping objects remain in the other-water output for review.\n'
        'Use q_area for area after clipping; static references do not establish reservoir construction dates.\n', encoding='utf-8')
    log(f'[SUCCESS] {args.output}')


if __name__ == '__main__':
    log('[START] QTP monthly reservoir separation')
    log(f'[SCRIPT] {Path(__file__).resolve()}')
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

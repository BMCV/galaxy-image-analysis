from .render import colorize_labels, normalize_image, render_ymap, render_result_over_image, render_atoms, render_adjacencies
from .batch import Task, _resolve_timings_key
from .output import get_output
from .io import imread, imwrite

import numpy as np
import gzip, dill, pathlib


DEFAULT_OUTDIR = {
    'seg': 'export-seg',
    'img': 'export-img',
    'fgc': 'export-fgc',
    'adj': 'export-adj',
    'atm': 'export-atm'
}

DEFAULT_BORDER = {
    'seg': 8,
    'fgc': 2,
    'adj': 2,
    'atm': 6,
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rootpath', help='root directory for batch processing')
    parser.add_argument('taskdir', help=f'batch task directory path')
    parser.add_argument('--outdir', help='output directory', default=None)
    parser.add_argument('--imageid', help='only export image ID', default=[], action='append')
    parser.add_argument('--border', help='border width', type=int, default=None)
    parser.add_argument('--border-position', help='border position (inner, center, outer)', type=str, default='center')
    parser.add_argument('--enhance', help='apply contrast enhancement', action='store_true')
    parser.add_argument('--mode', help='export the segmentation results (seg), the raw images (img), the foreground clusters (fgc), the adjacency graphs (adj), or the atoms (atm)', default='seg')
    parser.add_argument('--ymap', help='intensity mapping for y-map rendering', default='-0.8:+1:5:seismic')
    args = parser.parse_args()

    if args.mode not in ('seg', 'img', 'fgc', 'adj', 'atm'):
        parser.error(f'Unknown mode: "{args.mode}"')

    border_width = args.border
    if border_width is None and args.mode in DEFAULT_BORDER:
        border_width = DEFAULT_BORDER[args.mode]
        
    border_position = args.border_position

    if args.ymap.startswith('/'):
        args.ymap = args.ymap[1:]
    
    rootpath = pathlib.Path(args.rootpath)
    if not rootpath.exists():
        raise ValueError(f'Root path does not exist: {rootpath}')

    taskdir = pathlib.Path(args.taskdir)
    if not taskdir.is_absolute():
        taskdir = rootpath / taskdir
    if not taskdir.is_dir():
        raise ValueError(f'Task directory does not exist: {taskdir}')

    outdir = pathlib.Path(args.outdir if args.outdir is not None else DEFAULT_OUTDIR[args.mode])
    if not outdir.is_absolute():
        outdir = taskdir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    _taskdirs = [taskdir]
    while _taskdirs[-1] != rootpath:
        _taskdirs.append(_taskdirs[-1].parents[0])

    tasks = []
    for _taskdir in _taskdirs[::-1]:
        task = Task.create_from_directory(_taskdir, tasks[-1] if len(tasks) > 0 else None)
        if task is not None:
            tasks.append(task)
    task = tasks[-1]
    if not task.runnable:
        task = Task.create_from_directory(task.path, tasks[-2], force_runnable=True)

    out = get_output(None)
    if len(args.imageid) > 0:
        task.file_ids = [_resolve_timings_key(file_id, task.file_ids) for file_id in args.imageid]
    task.seg_pathpattern = None
    task.log_pathpattern = None
    task.adj_pathpattern = None
    task._load_timings = lambda *args: {}

    if args.mode == 'img':
        for image_id in task.file_ids:
            im_filepath = str(task.im_pathpattern) % image_id
            outputfile = outdir / f'{image_id}.png'
            out.intermediate(f'Processing image... {outputfile}')
            img = imread(im_filepath)
            if args.enhance: img = normalize_image(img)
            outputfile.parents[0].mkdir(parents=True, exist_ok=True)
            imwrite(str(outputfile), img)
    elif args.mode in ('seg', 'fgc', 'adj', 'atm'):
        if args.mode in ('fgc', 'adj', 'atm'):
            task.last_stage = 'c2f-region-analysis'
        if args.mode in ('fgc', 'adj'):
            ymap_spec = tuple(tf(val) for val, tf in zip(args.ymap.split(':'), (float, float, float, str)))
            ymapping  = lambda y: np.exp(ymap_spec[2] * y) / (1 + np.exp(ymap_spec[2] * y)) - 0.5
            render_ymap = lambda y: render_ymap(ymapping(y.clip(*ymap_spec[:2])), clim=(ymapping(np.array(ymap_spec[:2]))), cmap=ymap_spec[3])[:,:,:3]
            ymap_legend = render_ymap(np.linspace(*ymap_spec[:2], 200)[None, :])
            ymap_legend = np.vstack([ymap_legend] * 10)
            ymap_legendfile = outdir / f'ymap_legend.png'
            out.write(f'\nWriting legend: {ymap_legendfile}')
            imwrite(str(ymap_legendfile), ymap_legend)
        data = task.run(one_shot=True, force=True, evaluation='none', out=out)
        out.write('\nRunning export:')
        for image_id in task.file_ids:
            dataframe  = data[image_id]
            outputfile = outdir / f'{image_id}.png'
            out.intermediate(f'  Processing image... {outputfile}')
            outputfile.parents[0].mkdir(parents=True, exist_ok=True)
            if args.mode == 'seg':
                img = render_result_over_image(dataframe, border_width=border_width, border_position=border_position, normalize_img=args.enhance)
            elif args.mode == 'fgc':
                ymap = render_ymap(dataframe['y'])[:,:,:3]
                img  = render_foreground_clusters(dataframe, override_img=ymap, border_color=(0,0,0), border_radius=border_width // 2)
            elif args.mode == 'adj':
                ymap = render_ymap(dataframe['y'])[:,:,:3]
                ymap = render_atoms(dataframe, override_img=ymap, border_color=(0,0,0), border_radius=border_width // 2)
                img  = render_adjacencies(dataframe, override_img=ymap, edge_color=(0,1,0), endpoint_color=(0,1,0))
            elif args.mode == 'atm':
                img = render_atoms(dataframe, border_color=(0,1,0), border_radius=border_width // 2, normalize_img=args.enhance)
            imwrite(str(outputfile), img)
            out.write(f'  Exported {outputfile}')
        out.write('\n')
    out.write(f'Exported {len(task.file_ids)} files')


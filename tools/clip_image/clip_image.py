import giatools
import numpy as np

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.add_output_image('output')

    for section in tool.run(
        'YX',  # do not process all axes jointly so we can safely load a Dask array into a NumPy array (see below)
        output_dtype_hint='preserve',
    ):
        # Perform the clipping
        clip_args = [
            tool.args.params.get('lower_bound', -np.inf),
            tool.args.params.get('upper_bound', +np.inf),
        ]
        if clip_args == list(sorted(clip_args)):
            print('Applying clipping:', str(clip_args))
            section['output'] = np.asarray(  # conversion is required until https://github.com/BMCV/giatools/pull/44 is fixed
                section['input'].data.clip(*clip_args)
            )
        else:
            exit(f'Lower bound ({clip_args[0]:g}) must be less or equal compared to the upper bound ({clip_args[1]:g}).')

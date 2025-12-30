import giatools
import numpy as np
import pandas as pd
import skimage.measure

from giatools__0_7_2__cli__ToolBaseplate import parse_args  # noqa: I202


if __name__ == '__main__':
    tool = giatools.ToolBaseplate()
    tool.add_input_image('labels')
    tool.add_input_image('intensities', required=False)
    tool.parser.add_argument('--output', type=str)
    parse_args(tool)  # TODO: Revert to `tool.parse_args` when 0.7.2 is on Conda

    # Validate the input image
    try:
        label_image = tool.args.input_images['labels']
        if any(label_image.shape[label_image.axes.index(axis)] > 1 for axis in label_image.axes if axis not in 'ZYX'):
            raise ValueError(f'This tool is not applicable to images with {label_image.original_axes} axes.')

        # Extract the image features
        for section in tool.run('ZYX'):  # the validation code above guarantees that we will have only a single iteration
            df = pd.DataFrame()
            if (intensities_section := section['intensities']):
                regions = skimage.measure.regionprops(section['labels'].data, intensity_image=intensities_section.data)
            else:
                regions = skimage.measure.regionprops(section['labels'].data, intensity_image=None)

            df['it'] = np.arange(len(regions))

            for feature_name in tool.args.params['features']:
                if feature_name == 'label':
                    df['label'] = df['it'].map(lambda ait: regions[ait].label)
                elif feature_name == 'convexity':
                    perimeter = df['it'].map(lambda ait: regions[ait].perimeter)
                    area = df['it'].map(lambda ait: regions[ait].area)
                    df['convexity'] = area / (perimeter * perimeter)
                else:
                    try:
                        df[feature_name] = df['it'].map(lambda ait: getattr(regions[ait], feature_name))
                    except TypeError:
                        raise ValueError(f'Unknown feature: "{feature_name}"')

            del df['it']
            df.to_csv(tool.args.raw_args.output, sep='\t', lineterminator='\n', index=False)

    except ValueError as err:
        exit(err.args[0])

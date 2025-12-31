import giatools
import numpy as np
import pandas as pd

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('labels')
    tool.add_output_image('output')
    tool.parser.add_argument('--features', type=str, required=True)
    tool.parser.add_argument('--rules', type=str, required=True)
    tool.parse_args()

    features = pd.read_csv(tool.args.raw_args.features, delimiter='\t', index_col=0)
    features.columns = features.columns.str.strip()  # remove whitespaces from header names
    rules = pd.read_csv(tool.args.raw_args.rules, delimiter='\t')
    rules.columns = rules.columns.str.strip()  # remove whitespaces from header names

    # Write info to stdout that might be useful to the user if something is not working
    required_rules_columns = frozenset(('feature', 'min', 'max'))
    print('Features:', ', '.join(features.columns))
    if (spurious_rules_columns := frozenset(rules.columns) - required_rules_columns):
        print('Spurious rules columns:', ','.join(f'"{column}"' for column in spurious_rules_columns))

    # Validate the rules and the input image
    try:
        # Validate the rules
        if (missing_rules_columns := required_rules_columns - frozenset(rules.columns)):
            raise ValueError(f'Missing rules columns: {", ".join(missing_rules_columns)}')
        print('Rules for:', ', '.join(feature_name.strip() for feature_name in rules['feature']))

        # Validate the input image
        label_image = tool.args.input_images['labels']
        if any(label_image.shape[label_image.axes.index(axis)] > 1 for axis in label_image.axes if axis not in 'ZYX'):
            raise ValueError(f'This tool is not applicable to images with {label_image.original_axes} axes.')

        # Process all YX slices jointly (demands less memory when processing large Zarrs, as opposed to processing ZYX jointly)
        info_lines = list()  # list of lines to be printed to stdout when finished (do not print repeatedly on each iteration)
        for section in tool.run('YX', output_dtype_hint='preserve'):

            # Convert the slice to a NumPy array if it is a Dask array
            labels = section['labels'].data
            if hasattr(labels, 'compute'):
                labels = labels.compute()

            # Iterate all objects in the label map...
            result = labels.copy()
            for label in np.unique(labels):
                if label == 0:
                    continue  # skip the background

                mask = (labels == label)
                if label not in features.index:

                    # The label is not in the `features` file
                    if tool.args.params['missing'] == 'remove':
                        result[mask] = 0  # consider this as a manual removal
                    else:
                        pass  # keep the object and processed with the next label

                # Check the rules for the object
                else:
                    for rule in rules.to_dict(orient='records'):
                        feature_name = rule['feature'].strip()
                        if feature_name not in features.columns:
                            raise ValueError(f'Rule requires "{feature_name}" but this feature is missing.')
                        feature_value = features.loc[label, feature_name]

                        # Keep the object if it passes the rule
                        min_value = rule['min']
                        max_value = rule['max']
                        if feature_value < min_value or feature_value > max_value:
                            info_lines.append(
                                f'Remove object {label} due to {feature_name}={feature_value}, must be in [{min_value}, {max_value}]',
                            )
                            result[mask] = 0
                            break  # stop rule checking for current object, proceed with next label

            section['output'] = result
            for line in sorted(set(info_lines)):
                print(line)

    except ValueError as err:
        exit(err.args[0])

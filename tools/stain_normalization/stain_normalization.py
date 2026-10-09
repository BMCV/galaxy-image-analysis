import giatools

# Fail early if an optional backend is not available
giatools.require_backend('omezarr')


if __name__ == "__main__":
    tool = giatools.ToolBaseplate()
    tool.add_input_image('input')
    tool.add_output_image('output')

    for section in tool.run(
        'YXC',  # process the RGB image as a whole
        output_dtype_hint='preserve',
    ):
        section['output'] = section['input'].data

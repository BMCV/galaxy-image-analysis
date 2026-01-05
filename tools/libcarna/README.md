# libcarna

## HTML output

The tool produces HTML output that needs to be added to the allow list.

## GPU support

The following extra Docker parameters are required to run this tool with GPU support:
```bash
--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute
```
Otherwise, the tool runs with software rendering, whuch is much slower.

When using `planemo test`, the full command line is:
```bash
planemo test --docker --docker_run_extra_arguments \
    "--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute"
```

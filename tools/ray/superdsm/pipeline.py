from ._aux import copy_dict, mkdir
from .output import get_output
from .image import normalize_image

import math
import numpy as np
import time


class Stage(object):
    """A pipeline stage.

    Each stage can be controlled by a separate set of hyperparameters. Refer to the documentation of the respective pipeline stages for details. Most hyperparameters reside in namespaces, which are uniquely associated with the corresponding pipeline stages.

    :param name: Readable identifier of this stage.
    :param cfgns: Hyperparameter namespace of this stage. Defaults to ``name`` if not specified.
    :param inputs: List of inputs required by this stage.
    :param outputs: List of outputs produced by this stage.

    Automation
    ^^^^^^^^^^

    Hyperparameters can be set automatically using the :py:meth:`~.configure` method based on the scale :math:`\sigma` of objects in an image. Hyperparameters are only set automatically based on the scale of objects, if the :py:mod:`~superdsm.automation` module (as in :ref:`this <usage_example_interactive>` example) or batch processing are used (as in :ref:`this <usage_example_batch>` example). Hyperparameters are *not* set automatically if the :py:meth:`~superdsm.pipeline.Pipeline.process_image` method of the :py:class:`~superdsm.pipeline.Pipeline` class is used directly.

    Inputs and outputs
    ^^^^^^^^^^^^^^^^^^

    Each stage must declare its required inputs and the outputs it produces. These are used by :py:meth:`~.create_pipeline` to automatically determine the stage order. The input ``g_raw`` is provided by the pipeline itself.
    """

    def __init__(self, name, cfgns=None, inputs=[], outputs=[]):
        if cfgns is None: cfgns = name
        self.name    = name
        self.cfgns   = cfgns
        self.inputs  = dict([(key, key) for key in  inputs])
        self.outputs = dict([(key, key) for key in outputs])
        self._callbacks = {}

    def _callback(self, name, *args, **kwargs):
        if name in self._callbacks:
            for cb in self._callbacks[name]:
                cb(name, *args, **kwargs)

    def add_callback(self, name, cb):
        if name not in self._callbacks: self._callbacks[name] = []
        self._callbacks[name].append(cb)

    def remove_callback(self, name, cb):
        if name in self._callbacks: self._callbacks[name].remove(cb)

    def __call__(self, data, cfg, out=None, log_root_dir=None):
        out = get_output(out)
        cfg = cfg.get(self.cfgns, {})
        if cfg.get('enabled', self.ENABLED_BY_DEFAULT):
            out.intermediate(f'Starting stage "{self.name}"')
            self._callback('start', data)
            input_data = {}
            for data_key, input_data_key in self.inputs.items():
                input_data[input_data_key] = data[data_key]
            t0 = time.time()
            output_data = self.process(input_data, cfg=cfg, out=out, log_root_dir=log_root_dir)
            dt = time.time() - t0
            assert len(set(output_data.keys()) ^ set(self.outputs)) == 0, 'stage "%s" generated unexpected output' % self.name
            for output_data_key, data_key in self.outputs.items():
                data[data_key] = output_data[output_data_key]
            self._callback('end', data)
            return dt
        else:
            out.write(f'Skipping disabled stage "{self.name}"')
            self._callback('skip', data)
            return 0

    def process(self, input_data, cfg, out, log_root_dir):
        """Runs this pipeline stage.

        :param input_data: Dictionary of the inputs declared by this stage.
        :param cfg: The hyperparameters to be used by this stage.
        :param out: An instance of an :py:class:`~superdsm.output.Output` sub-class, ``'muted'`` if no output should be produced, or ``None`` if the default output should be used.
        :param log_root_dir: Path of directory where log files will be written, or ``None`` if no log files should be written.
        :return: Dictionary of the outputs declared by this stage.
        """
        raise NotImplementedError()

    def configure(self, scale):
        radius   = scale * math.sqrt(2)
        diameter = 2 * radius
        return self.configure_ex(scale, radius, diameter)

    def configure_ex(self, scale, radius, diameter):
        """Automatically computes the default configuration entries which are dependent on the scale of the objects in an image, using explicit values for the expected radius and diameter of the objects.

        :param scale: The average scale of objects in the image.
        :param radius: The average radius of objects in the image.
        :param diameter: The average diameter of objects in the image.
        :return: Dictionary of configuration entries of the form:

            .. code-block:: python

               {
                   'key': (factor, default_user_factor),
               }
            
            Each hyperparameter ``key`` is associated with a new hyperparameter ``AF_key``. The value of the hyperparameter ``key`` will be computed as the product of ``factor`` and the value of the ``AF_key`` hyperparameter, which defaults to ``default_user_factor``. The value given for ``factor`` is usually ``scale``, ``radius``, ``diameter``, or a polynomial thereof. Another dictionary may be provided as a third component of the tuple, which can specify a ``type``, ``min``, and ``max`` values.
        """
        return dict()


class ProcessingControl:

    def __init__(self, first_stage=None, last_stage=None):
        self.started     = True if first_stage is None else False
        self.first_stage = first_stage
        self.last_stage  =  last_stage
    
    def step(self, stage):
        if not self.started and stage == self.first_stage: self.started = True
        do_step = self.started
        if stage == self.last_stage: self.started = False
        return do_step


class Pipeline:
    """Represents a processing pipeline for image segmentation.
    
    Note that hyperparameters are *not* set automatically if the :py:meth:`~.process_image` method is used directly. Hyperparameters are only set automatically based on the scale of objects, if the :py:mod:`~superdsm.automation` module (as in :ref:`this <usage_example_interactive>` example) or batch processing are used (as in :ref:`this <usage_example_batch>` example). 
    """
    
    def __init__(self):
        self.stages = []

    def process_image(self, g_raw, cfg, first_stage=None, last_stage=None, data=None, out=None, log_root_dir=None):
        """Performs the segmentation of an image.

        First, the image is provided to the stages of the pipeline using the :py:meth:`.init` method. Then, the :py:meth:`~.Stage.process` methods of the stages of the pipeline are executed successively.

        :param g_raw: A ``numpy.ndarray`` object corresponding to the image which is to be processed.
        :param cfg: A :py:class:`~superdsm.config.Config` object which represents the hyperparameters.
        :param first_stage: The name of the first stage to be executed.
        :param last_stage: The name of the last stage to be executed.
        :param data: The results of a previous execution.
        :param out: An instance of an :py:class:`~superdsm.output.Output` sub-class, ``'muted'`` if no output should be produced, or ``None`` if the default output should be used.
        :param log_root_dir: Path to a directory where log files should be written to.
        :return: Tuple ``(data, cfg, timings)``, where ``data`` is the *pipeline data object* comprising all final and intermediate results, ``cfg`` are the finally used hyperparameters, and ``timings`` is a dictionary containing the execution time of each individual pipeline stage (in seconds).

        The parameter ``data`` is used if and only if ``first_stage`` is not ``None``. In this case, the outputs produced by the stages of the pipeline which are being skipped must be fed in using the ``data`` parameter obtained from a previous execution of this method.
        """
        cfg = cfg.copy()
        if log_root_dir is not None: mkdir(log_root_dir)
        if first_stage == self.stages[0].name and data is None: first_stage = None
        if first_stage is not None and first_stage.endswith('+'): first_stage = self.stages[1 + self.find(first_stage[:-1])].name
        if first_stage is not None and last_stage is not None and self.find(first_stage) > self.find(last_stage): return data, cfg, {}
        out  = get_output(out)
        ctrl = ProcessingControl(first_stage, last_stage)
        if ctrl.step('init'): data = self.init(g_raw, cfg)
        else: assert data is not None, 'data argument must be provided if first_stage is used'
        timings = {}
        for stage in self.stages:
            if ctrl.step(stage.name):
                dt = stage(data, cfg, out=out, log_root_dir=log_root_dir)
                timings[stage.name] = dt
        return data, cfg, timings

    def init(self, g_raw, cfg):
        """Initializes the pipeline for processing the image ``g_raw``.

        :param g_raw: The image which is to be processed by the pipeline.
        :param cfg: The hyperparameters.

        The image ``g_raw`` is made available as an input to the pipeline stages. However, if ``cfg['histological'] == True`` (i.e. the hyperparameter ``histological`` is set to ``True``), then ``g_raw`` is converted to a brightness-inverse intensity image, and the original image is provided as ``g_rgb`` to the stages of the pipeline.

        In addition, ``g_raw`` is normalized so that the intensities range from 0 to 1.
        """
        if cfg.get('histological', False):
            g_rgb = g_raw
            g_raw = g_raw.mean(axis=2)
            g_raw = g_raw.max() - g_raw
        else:
            g_rgb = None
        data = dict(g_raw = normalize_image(g_raw))
        if g_rgb is not None:
            data['g_rgb'] = g_rgb
        return data

    def find(self, stage_name, not_found_dummy=np.inf):
        """Returns the position of the stage identified by ``stage_name``.

        Returns ``not_found_dummy`` if the stage is not found.
        """
        try:
            return [stage.name for stage in self.stages].index(stage_name)
        except ValueError:
            return not_found_dummy

    def append(self, stage, after=None):
        if after is None: self.stages.append(stage)
        else:
            if isinstance(after, str): after = self.find(after)
            self.stages.insert(after + 1, stage)


def create_pipeline(stages):
    """Creates and returns a new :py:class:`.Pipeline` object configured for the given stages.

    The stage order is determined automatically.
    """
    available_inputs = set(['g_raw'])
    remaining_stages = list(stages)

    pipeline = Pipeline()
    while len(remaining_stages) > 0:
        next_stage = None
        for stage in remaining_stages:
            if frozenset(stage.inputs.keys()).issubset(available_inputs):
                next_stage = stage
                break
        if next_stage is None:
            raise ValueError('failed to resolve total ordering')
        remaining_stages.remove(next_stage)
        pipeline.append(next_stage)
        available_inputs |= frozenset(next_stage.outputs.keys())

    return pipeline


def create_default_pipeline():
    from .c2freganal import C2F_RegionAnalysis

    stages = [
        C2F_RegionAnalysis(),
    ]

    return create_pipeline(stages)


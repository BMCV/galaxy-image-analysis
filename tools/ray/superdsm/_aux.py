import numpy as np
import scipy.sparse
import warnings
import pathlib
import ray
import fcntl, hashlib
import posix_ipc


def copy_dict(d):
    """Returns a copy of dict `d`.
    """
    assert isinstance(d, dict), 'not a "dict" object'
    return {item[0]: copy_dict(item[1]) if isinstance(item[1], dict) else item[1] for item in d.items()}


def uplift_smooth_matrix(smoothmat, mask):
    assert mask.sum() == smoothmat.shape[0], 'smooth matrix and region mask are incompatible'
    if not scipy.sparse.issparse(smoothmat): warnings.warn(f'{uplift_smooth_matrix.__name__} received a dense matrix which is inefficient')
    M = scipy.sparse.coo_matrix((np.prod(mask.shape), smoothmat.shape[0]))
    M.data = np.ones(mask.sum())
    M.row  = np.where(mask.reshape(-1))[0]
    M.col  = np.arange(len(M.data))
    smoothmat2 = M.tocsr() @ smoothmat
    return smoothmat2


def mkdir(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def join_path(path1, path2):
    return str(pathlib.Path(path1) / pathlib.Path(path2))


def is_subpath(path, subpath):
    if isinstance(   path, str):    path = pathlib.Path(   path)
    if isinstance(subpath, str): subpath = pathlib.Path(subpath)
    try:
        subpath.relative_to(path)
        return True
    except ValueError:
        return False


def get_ray_1by1(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids, num_returns=1)
        assert len(done) == 1
        yield ray.get(done[0])


def render_objects_foregrounds(shape, objects):
    foreground = np.zeros(shape, bool)
    for obj in objects:
        sel = obj.fill_foreground(foreground)
        yield foreground
        foreground[sel].fill(False)


class SystemSemaphore:
    def __init__(self, name, limit):
        self.name  = name
        self.limit = limit

    @staticmethod
    def get_lock(lock):
        class NullLock:
            def __enter__(self): pass
            def __exit__ (self, _type, value, tb): pass
        if lock is None: return NullLock()
        else: return lock

    def __enter__(self):
        if np.isinf(self.limit):
            create_lock = lambda flags: None
        else:
            create_lock = lambda flags: posix_ipc.Semaphore(f'/{self.name}', flags, mode=384, initial_value=self.limit)
        self._lock = create_lock(posix_ipc.O_CREAT | posix_ipc.O_EXCL)
        class Lock:
            def __init__(self, create_lock):
                self._create_lock = create_lock

            def __enter__(self):
                self._lock = self._create_lock(posix_ipc.O_CREAT)
                if self._lock is not None: self._lock.acquire()

            def __exit__(self, _type, value, tb):
                if self._lock is not None: self._lock.release()
        return Lock(create_lock)

    def __exit__(self, _type, value, tb):
        if self._lock is not None: self._lock.unlink()


class SystemMutex:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        lock_id = hashlib.md5(self.name.encode('utf8')).hexdigest()
        self.fp = open(f'/tmp/.lock-{lock_id}.lck', 'wb')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
        
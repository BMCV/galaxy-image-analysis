try:
    from ._libs.sparse_dot_mkl import dot_product_mkl as dot, dot_product_transpose_mkl as gram

except ImportError:
    def stub(*args, **kwargs): raise NotImplementedError('Failed to load MKL')
    dot  = stub
    gram = stub


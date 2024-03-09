import argparse
import ast
import operator

import numpy as np
import skimage.io


supported_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


supported_functions = {
    'sqrt': np.sqrt,
    'abs': abs,
}


def eval_ast_node(node, inputs):
    """
    Evaluates a node of the syntax tree.
    """

    # Numeric constants evaluate to numeric values.
    if isinstance(node, ast.Constant):
        assert type(node.value) in (int, float)
        return node.value

    # Variables are looked up from the inputs and resolved.
    if isinstance(node, ast.Name):
        assert node.id in inputs.keys()
        return inputs[node.id]

    # Binary operators are evaluated based on the `supported_operators` dictionary.
    if isinstance(node, ast.BinOp):
        assert type(node.op) in supported_operators.keys(), node.op
        op = supported_operators[type(node.op)]
        return op(eval_ast_node(node.left, inputs), eval_ast_node(node.right, inputs))

    # Unary operators are evaluated based on the `supported_operators` dictionary.
    if isinstance(node, ast.UnaryOp):
        assert type(node.op) in supported_operators.keys(), node.op
        op = supported_operators[type(node.op)]
        return op(eval_ast_node(node.operand, inputs))

    # Function calls are evaluated based on the `supported_functions` dictionary.
    if isinstance(node, ast.Call):
        assert len(node.args) == 1 and len(node.keywords) == 0
        assert node.func.id in supported_functions.keys(), node.func.id
        func = supported_functions[node.func.id]
        return func(eval_ast_node(node.args[0], inputs))

    # The node is unsupported and could not be evaluated.
    raise TypeError(f'Unsupported node type: "{node}"')


def eval_expression(expr, inputs):
    return eval_ast_node(ast.parse(expr, mode='eval').body, inputs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--expression', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', default=list(), action='append', required=True)
    args = parser.parse_args()

    inputs = dict()
    im_shape = None
    for input in args.input:
        name, filepath = input.split(':')
        im = skimage.io.imread(filepath)
        assert name not in inputs, 'Input name "{name}" is ambiguous.'
        inputs[name] = im
        if im_shape is None:
            im_shape = im.shape
        else:
            assert im.shape == im_shape, 'Input images differ in size and/or number of channels.'

    result = eval_expression(args.expression, inputs)

    skimage.io.imsave(args.output, result)

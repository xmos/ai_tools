import io
import tensorflow as tf
from contextlib import redirect_stdout


def get_operator_counts(model_content):
    with io.StringIO() as buf, redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_content=model_content)
        model_structure = buf.getvalue()

    operators = [
        op.strip().split(" ")[1].split("(")[0]
        for op in model_structure.split("\n")
        if "Op#" in op
    ]
    op_counts = {}
    for operator in operators:
        if operator in op_counts:
            op_counts[operator] = op_counts[operator] + 1
        else:
            op_counts[operator] = 1

    return (len(operators), op_counts)


def print_operator_counts(model_content):
    total_op_count, op_counts = get_operator_counts(model_content)
    print(f"{'Operator'.upper():<20} {'Count'.upper():>6}")
    print("-" * 20 + " " + "-" * 6)

    for operator, count in op_counts.items():
        print(f"{operator.lower():<20} {count:>6}")

    print("-" * 20 + " " + "-" * 6)
    print(f"{'Total'.upper():<20} {total_op_count:>6}")
    print("-" * 20 + " " + "-" * 6)

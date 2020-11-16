# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from typing import Type

from tflite2xcore.execution_planning import ExecutionPlanner, ReverseDepthFirstPlanner

from tflite2xcore import xcore_schema as xir

DUMMY_OPERATOR_CODE = xir.OperatorCode(xir.XCOREOpCodes.DUMMY)


@pytest.fixture()  # type: ignore
def PlannerUnderTest() -> Type[ReverseDepthFirstPlanner]:
    return ReverseDepthFirstPlanner


def test_single_op_with_const(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )
    tconst = subgraph.create_tensor("const", tin.type, tin.shape)
    tout = subgraph.create_tensor("output", tin.type, tin.shape, isoutput=True)
    op = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[tin, tconst], outputs=[tout]
    )

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op]


def test_single_op_with_two_outputs(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )
    tout1 = subgraph.create_tensor("output1", tin.type, tin.shape, isoutput=True)
    tout2 = subgraph.create_tensor("output2", tin.type, tin.shape, isoutput=True)
    op = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[tout1, tout2]
    )

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op]


def test_linear_graph(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )

    t1 = subgraph.create_tensor("intermediate1", tin.type, tin.shape)
    op1 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[t1])

    t2 = subgraph.create_tensor("intermediate2", t1.type, t1.shape)
    op2 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[t1], outputs=[t2])

    tout = subgraph.create_tensor("output", t2.type, t2.shape, isoutput=True)
    op3 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[t2], outputs=[tout])

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op1, op2, op3]


def test_order_by_size(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )

    tout1 = subgraph.create_tensor("output1", tin.type, (1,), isoutput=True)
    op1 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[tout1])

    tout2 = subgraph.create_tensor("output2", tin.type, (3,), isoutput=True)
    op2 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[tout2])

    tout3 = subgraph.create_tensor("output3", tin.type, (2,), isoutput=True)
    op3 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[tout3])

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op1, op3, op2]


def test_symmetric_parallel_block(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )
    t1_in = subgraph.create_tensor("branch1_input", tin.type, tin.shape)
    t2_in = subgraph.create_tensor("branch2_input", tin.type, tin.shape)
    op0 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[t1_in, t2_in]
    )

    t1_out = subgraph.create_tensor("branch1_output", t1_in.type, (2,))  # bigger
    op1 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[t1_in], outputs=[t1_out]
    )

    t2_out = subgraph.create_tensor("branch2_output", t1_out.type, (1,))  # smaller
    op2 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[t2_in], outputs=[t2_out]
    )

    tout = subgraph.create_tensor("output", tin.type, tin.shape, isoutput=True)
    op3 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[t1_out, t2_out], outputs=[tout]
    )

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op0, op2, op1, op3]


def test_asymmetric_parallel_block(PlannerUnderTest: Type[ExecutionPlanner]) -> None:
    model = xir.XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", xir.TensorType.FLOAT32, shape=(1,), isinput=True
    )
    tmid = subgraph.create_tensor("branch_input", tin.type, tin.shape)
    op0 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tin], outputs=[tmid])

    t1_mid = subgraph.create_tensor("branch1_mid", tmid.type, tmid.shape)
    op1 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tmid], outputs=[t1_mid])

    t1_out = subgraph.create_tensor("branch1_out", t1_mid.type, shape=(1,))  # smaller
    op2 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[t1_mid], outputs=[t1_out]
    )

    t2_out = subgraph.create_tensor("branch2_out", tmid.type, shape=(2,))  # bigger
    op3 = subgraph.create_operator(DUMMY_OPERATOR_CODE, inputs=[tmid], outputs=[t2_out])

    tout = subgraph.create_tensor("output", tin.type, tin.shape, isoutput=True)
    op4 = subgraph.create_operator(
        DUMMY_OPERATOR_CODE, inputs=[t1_out, t2_out], outputs=[tout]
    )

    planner = PlannerUnderTest(subgraph)
    assert planner.make_plan() == [op0, op1, op2, op3, op4]


if __name__ == "__main__":
    pytest.main()

from smolagents import Tool
from decimal import Decimal


class SumNumTool(Tool):
    name = "sum_two_numbers"
    description = "get the sum of two numbers"
    inputs = {
        "n1": {"type": "number", "description": "The first number"},
        "n2": {"type": "number", "description": "The second number"},
    }
    output_type = "number"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, n1, n2):
        return sum([Decimal(str(n1)), Decimal(str(n2))])


# must have var tools of classes
tools = [SumNumTool]

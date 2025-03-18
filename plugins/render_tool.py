from smolagents import Tool


class RenderBase64ImageTool(Tool):
    name = "render_base64_image"
    description = "convert base64 image to markdown syntax, then you can render it with final_answer tool"
    inputs = {
        "base64_value": {"type": "string", "description": "The encoded base64 value"},
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base64_value: str) -> str:
        return "![Image](data:image/png;base64,{})".format(base64_value)


# must have var tools of classes
tools = [RenderBase64ImageTool]

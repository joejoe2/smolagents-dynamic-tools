from smolagents import Tool
import time


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"
    default_kwargs = {"sleep_limit": 60}

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = {**self.default_kwargs, **kwargs}
        self.sleep_limit = (
            self.default_kwargs["sleep_limit"]
            if kwargs.get("sleep_limit") is None
            else kwargs["sleep_limit"]
        )
        self.is_waiting = False
        self.user_input = ""
        self.question = ""

    def forward(self, question):
        self.question = question
        self.is_waiting = True
        print("waiting for user's input...")
        sleep_count = 0
        while self.is_waiting:
            time.sleep(1)
            sleep_count += 1
            if sleep_count > self.sleep_limit:
                self.is_waiting = False
                return (
                    "User is absent now, cannot provide any input."
                    "Please terminate current task and write down the reason in final answer !"
                )
        return self.user_input

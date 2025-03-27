import json
import threading
from typing import Optional

import gradio as gr
from PIL.Image import Image
from requests import session
from smolagents import (
    # stream_to_gradio,
    populate_template,
    OpenAIServerModel,
    CodeAgent,
    ActionStep,
    TaskStep,
    handle_agent_output_types,
    AgentText,
    AgentImage,
    AgentAudio,
)
from smolagents.gradio_ui import pull_messages_from_step

from utils import load_tools

update_tool_prompt_prefix = """
Forget all previous tools. You are now working with a new set of tools, and you must only focus on these tools to assist with the task at hand.
Do **not** reference any previous tools, and only consider the current tools provided below.
Here are the tools currently available:"""
update_tool_prompt_template = (
    update_tool_prompt_prefix
    + """
  {%- for tool in tools.values() %}
  - {{ tool.name }}: {{ tool.description }}
      Takes inputs: {{tool.inputs}}
      Returns an output of type: {{tool.output_type}}
  {%- endfor %}
"""
)


def clean_memory_to_save_tokens(agent: CodeAgent):
    # modify memory
    for previous_memory_step in agent.memory.steps:
        if isinstance(previous_memory_step, TaskStep):
            # remove update_tool_prompt to save tokens
            pos = previous_memory_step.task.rfind(update_tool_prompt_prefix)
            if pos != -1:
                previous_memory_step.task = previous_memory_step.task[:pos].strip()


def stream_to_gradio(
    agent: CodeAgent,
    task: str,
    reset_agent_memory: bool = False,
    images: list[Image] = None,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(
        task,
        stream=True,
        reset=reset_agent_memory,
        images=images,
        additional_args=additional_args,
    ):
        # Track tokens if model provides them
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(
            role="assistant", content=f"**Final answer:** {str(final_answer)}"
        )


class MyGradioUI:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        authorized_imports: list[str],
        session_ttl: int = 3600,
        session_capacity: int = 10000,
    ):
        self.api_base: str = api_base
        self.api_key: str = api_key
        self.model: str = model
        self.authorized_imports: list[str] = authorized_imports
        self.session_ttl: int = session_ttl
        self.session_capacity: int = session_capacity

    def init_session(self, session):
        model = OpenAIServerModel(
            model_id=self.model, api_base=self.api_base, api_key=self.api_key
        )

        session["agent"] = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=self.authorized_imports,
            verbosity_level=2,
        )
        session["available_tools"] = load_tools()
        session["lock"] = threading.Lock()
        return (
            gr.Dropdown(choices=list(session["available_tools"].keys())),
            gr.Button(interactive=True),
            session,
        )

    def update_selected_tools(self, options: list[str], session):
        with session["lock"]:
            # delete removed tools
            old_tools = set(session["agent"].tools.keys())
            for old_tool in old_tools:
                if old_tool != "final_answer" and old_tool not in options:
                    del session["agent"].tools[old_tool]
            # add selected tools
            for option in options:
                if option in session["available_tools"] and option not in old_tools:
                    tool_cls = session["available_tools"][option]
                    try:
                        session["agent"].tools[tool_cls.name] = tool_cls()
                    except Exception as e:
                        raise Exception("Cannot create {} with {}".format(tool_cls, e))
            print("udpate tools: {}".format(session["agent"].tools))
            tools = list(session["agent"].tools.keys())
            tools.remove("final_answer")
            return gr.Dropdown(
                choices=tools,
                value=None,
                label="Tool Kwargs",
            )

    def reload_tools(self, options: list[str], session):
        with session["lock"]:
            # reload tools
            session["available_tools"] = load_tools()
            # clear
            old_tools = list(session["agent"].tools.keys())
            for old_tool in old_tools:
                if old_tool != "final_answer":
                    del session["agent"].tools[old_tool]
            # recreate selected tools
            for option in options:
                if option in session["available_tools"]:
                    tool_cls = session["available_tools"][option]
                    try:
                        session["agent"].tools[tool_cls.name] = tool_cls()
                    except Exception as e:
                        raise Exception("Cannot create {} with {}".format(tool_cls, e))
            print("reload tools: {}".format(session["agent"].tools))
            # rerun updated options
            tools = list(session["agent"].tools.keys())
            tools.remove("final_answer")
            return gr.Dropdown(
                choices=list(session["available_tools"].keys())
            ), gr.Dropdown(
                choices=tools,
                value=None,
                label="Tool Kwargs",
            )

    def update_tool_kwargs(self, option: str, kwargs_str, session):
        if not option:
            return (
                gr.Button("Update Kwargs"),
                gr.Dropdown(label="Tool Kwargs"),
                gr.Code(value="{}", language="json", interactive=False),
            )

        with session["lock"]:
            if option in session["available_tools"]:
                tool_cls = session["available_tools"][option]
                tool = session["agent"].tools[option]
                # recreate tool
                if hasattr(tool, "kwargs") and hasattr(tool_cls, "default_kwargs"):
                    try:
                        kwargs = json.loads(kwargs_str)
                        new_tool = tool_cls(**kwargs)
                        session["agent"].tools[tool_cls.name] = new_tool
                        return (
                            gr.Button("Update Kwargs"),
                            gr.Dropdown(label="Tool Kwargs"),
                            gr.Code(
                                value=json.dumps(kwargs, indent=4),
                                language="json",
                                interactive=True,
                            ),
                        )
                    except Exception as e:
                        raise Exception(
                            "Cannot create {} with {}, {}".format(
                                tool_cls, kwargs_str, e
                            )
                        )

        return (
            gr.Button("Update Kwargs"),
            gr.Dropdown(label="Tool Kwargs"),
            gr.Code(value="{}", language="json"),
        )

    def update_selected_kwargs_input(self, option: str, session):
        if not option:
            return gr.Code(value="{}", language="json", interactive=False)

        with session["lock"]:
            if option in session["available_tools"]:
                tool_cls = session["available_tools"][option]
                tool = session["agent"].tools[option]
                default_kwargs = getattr(tool_cls, "default_kwargs", None)
                kwargs = getattr(tool, "kwargs", None)
                if kwargs is not None and default_kwargs is not None:
                    return gr.Code(
                        value=json.dumps(kwargs, indent=4),
                        language="json",
                        interactive=True,
                    )
        return gr.Code(value="{}", language="json")

    def upload_image(self, image: Image, session):
        with session["lock"]:
            # pil image or None
            if image is None:
                session["image_input"] = None
            else:
                session["image_input"] = image

    def interact_with_agent(self, prompt, messages, session_state):
        try:
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                raise ValueError("Cannot interact with empty prompt !")
            with session_state["lock"]:
                # save tokens
                clean_memory_to_save_tokens(session_state["agent"])

                # display messages to user
                messages.append(gr.ChatMessage(role="user", content=prompt))
                yield messages

                # fix agent tools update problem
                update_tool_prompt = populate_template(
                    update_tool_prompt_template,
                    variables={"tools": session_state["agent"].tools},
                )
                prompt = prompt + "\n\n" + update_tool_prompt

                # get image input
                images: list[Image] = (
                    [session_state["image_input"].copy()]
                    if (
                        "image_input" in session_state
                        and isinstance(session_state["image_input"], Image)
                    )
                    else None
                )

                for msg in stream_to_gradio(
                    session_state["agent"],
                    task=prompt,
                    reset_agent_memory=False,
                    images=images,
                ):
                    messages.append(msg)
                    yield messages

                yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e.with_traceback())}")
            messages.append(
                gr.ChatMessage(role="assistant", content=f"Error: {str(e)}")
            )
            yield messages

    def launch(self, **kwargs):
        with gr.Blocks(
            theme="ocean", fill_height=True, css="footer{display:none !important}"
        ) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({}, time_to_live=self.session_ttl)

            with gr.Tab("Agent"):
                # main chat interface
                chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(
                        None,
                        None,
                    ),
                    resizeable=True,
                )

                with gr.Row():
                    # chat input
                    text_input = gr.Textbox(
                        lines=2,
                        scale=10,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here.",
                    )
                    buffer_input = gr.Textbox(interactive=False, visible=False)
                    with gr.Column(scale=1, min_width=20):
                        # submit btn
                        submit_btn = gr.Button(
                            ">", variant="primary", interactive=False
                        )
                        # image input
                        image_input = gr.Image(
                            format="png",
                            type="pil",
                            sources=["upload"],
                            label="",
                            placeholder="",
                            height=30,
                        )

            with gr.Tab("Setting"):
                # dynamically tool selection
                tool_selections = gr.Dropdown(
                    choices=[], label="Tools", multiselect=True
                )

                # customize tool kwargs
                tool_kwargs_selections = gr.Dropdown(choices=[], label="Tool Kwargs")
                tool_kwargs_input = gr.Code(value="{}", language="json")
                tool_kwargs_submit = gr.Button("Update Kwargs")

                # reload tools
                reload_tool_btn = gr.Button("Reload Tools")

            # Set up event handlers
            text_input.submit(
                lambda x: (
                    x,
                    gr.Textbox(value="", interactive=False),
                    gr.Button(interactive=False),
                ),
                [text_input],
                [buffer_input, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [buffer_input, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )
            image_input.change(self.upload_image, inputs=[image_input, session_state])

            submit_btn.click(
                lambda x: (
                    x,
                    gr.Textbox(value="", interactive=False),
                    gr.Button(interactive=False),
                ),
                [text_input],
                [buffer_input, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [buffer_input, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            tool_selections.change(
                fn=self.update_selected_tools,
                inputs=[tool_selections, session_state],
                outputs=[tool_kwargs_selections],
            )
            tool_kwargs_selections.change(
                fn=self.update_selected_kwargs_input,
                inputs=[tool_kwargs_selections, session_state],
                outputs=[tool_kwargs_input],
            )
            tool_kwargs_submit.click(
                fn=self.update_tool_kwargs,
                inputs=[tool_kwargs_selections, tool_kwargs_input, session_state],
                outputs=[tool_kwargs_submit, tool_kwargs_selections, tool_kwargs_input],
            )
            reload_tool_btn.click(
                self.reload_tools,
                inputs=[tool_selections, session_state],
                outputs=[tool_selections, tool_kwargs_selections],
            )

            # init session agent and tools, only set session_state as output here for auto timeout
            demo.load(
                self.init_session,
                inputs=[session_state],
                outputs=[tool_selections, submit_btn, session_state],
            )

        demo.launch(debug=True, state_session_capacity=self.session_capacity, **kwargs)

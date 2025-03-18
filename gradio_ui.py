import json
import threading

import gradio as gr
from smolagents import (
    stream_to_gradio,
    populate_template,
    OpenAIServerModel,
    CodeAgent,
    ActionStep,
    TaskStep,
)

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


class MyGradioUI:
    def __init__(
        self, api_base: str, api_key: str, model: str, authorized_imports: list[str]
    ):
        self.api_base: str = api_base
        self.api_key: str = api_key
        self.model: str = model
        self.authorized_imports: list[str] = authorized_imports

    def init_session(self, session):
        model = OpenAIServerModel(
            model_id=self.model, api_base=self.api_base, api_key=self.api_key
        )

        def update_memory(memory_step: ActionStep, agent: CodeAgent):
            for previous_memory_step in agent.memory.steps:
                if isinstance(previous_memory_step, TaskStep):
                    # remove update_tool_prompt to save tokens
                    pos = previous_memory_step.task.rfind(update_tool_prompt_prefix)
                    if pos != -1:
                        previous_memory_step.task = previous_memory_step.task[
                            :pos
                        ].strip()

        session["agent"] = CodeAgent(
            tools=[],
            model=model,
            verbosity_level=2,
            additional_authorized_imports=self.authorized_imports,
            step_callbacks=[update_memory],
        )
        session["available_tools"] = load_tools()
        session["lock"] = threading.Lock()
        return gr.Dropdown(choices=list(session["available_tools"].keys())), gr.Button(
            interactive=True
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
                        print("Cannot create {} with {}".format(tool_cls, e))
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
                        print("Cannot create {} with {}".format(tool_cls, e))
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
                        print(
                            "Cannot create {} with {}, {}".format(
                                tool_cls, kwargs_str, e
                            )
                        )
                        return (
                            gr.Button("Update Kwargs"),
                            gr.Dropdown(label="Tool Kwargs"),
                            gr.Code(
                                value=json.dumps(tool.kwargs, indent=4),
                                language="json",
                                interactive=True,
                            ),
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

    def interact_with_agent(self, prompt, messages, session_state):
        try:
            with session_state["lock"]:
                messages.append(gr.ChatMessage(role="user", content=prompt))
                yield messages

                # fix agent tools update problem
                update_tool_prompt = populate_template(
                    update_tool_prompt_template,
                    variables={"tools": session_state["agent"].tools},
                )
                prompt = prompt + "\n\n" + update_tool_prompt
                for msg in stream_to_gradio(
                    session_state["agent"], task=prompt, reset_agent_memory=False
                ):
                    messages.append(msg)
                    yield messages

                yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            messages.append(
                gr.ChatMessage(role="assistant", content=f"Error: {str(e)}")
            )
            yield messages

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, **kwargs):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
            )
            text_input = gr.Textbox(
                lines=2,
                label="Chat Message",
                container=False,
                placeholder="Enter your prompt here.",
            )
            submit_btn = gr.Button("Submit", variant="primary", interactive=False)

            # Set up event handlers
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True, placeholder="Enter your prompt here."),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(interactive=True, placeholder="Enter your prompt here."),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            # dynamically tool selection
            tool_selections = gr.Dropdown(choices=[], label="Tools", multiselect=True)

            # customize tool kwargs
            tool_kwargs_selections = gr.Dropdown(choices=[], label="Tool Kwargs")
            tool_kwargs_input = gr.Code(value="{}", language="json")
            tool_kwargs_submit = gr.Button("Update Kwargs")

            # reload tools
            reload_tool_btn = gr.Button("Reload Tools")

            # events
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

            # init session agent and tools
            demo.load(
                self.init_session,
                inputs=[session_state],
                outputs=[tool_selections, submit_btn],
            )

        demo.launch(debug=True, **kwargs)

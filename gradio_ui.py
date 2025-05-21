import json
import threading

import gradio as gr
from PIL.Image import Image
from gradio_modal import Modal
from smolagents import (
    stream_to_gradio,
    populate_template,
    OpenAIServerModel,
    CodeAgent,
    TaskStep,
)

from prompt import (
    UPDATE_TOOL_PROMPT_PREFIX,
    UPDATE_TOOL_PROMPT_TEMPLATE,
    TASK_PRE_PROMPT,
)
from user_input_tool import UserInputTool
from utils import load_tools


def clean_memory_to_save_tokens(agent: CodeAgent):
    # modify memory
    for previous_memory_step in agent.memory.steps:
        if isinstance(previous_memory_step, TaskStep):
            # remove TASK_PRE_PROMPT to save tokens
            cleaned = previous_memory_step.task.replace(TASK_PRE_PROMPT, "")
            # remove UPDATE_TOOL_PROMPT to save tokens
            pos = cleaned.rfind(UPDATE_TOOL_PROMPT_PREFIX)
            if pos != -1:
                cleaned = cleaned[:pos]
            previous_memory_step.task = cleaned.strip()


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
                        gr.Warning("Cannot create {} with {}".format(tool_cls, e))
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
                        gr.Warning("Cannot create {} with {}".format(tool_cls, e))
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
                        raise gr.Warning(
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
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            gr.Warning("Cannot interact with empty prompt !")
            return messages
        try:
            with session_state["lock"]:
                # save tokens
                clean_memory_to_save_tokens(session_state["agent"])

                # display messages to user
                messages.append(gr.ChatMessage(role="user", content=prompt))
                yield messages

                # add problem tag
                prompt = "Problem: " + prompt

                # fix agent tools update problem
                update_tool_prompt = populate_template(
                    UPDATE_TOOL_PROMPT_TEMPLATE,
                    variables={"tools": session_state["agent"].tools},
                )
                prompt = prompt + "\n\n" + update_tool_prompt

                # add TASK_PRE_PROMPT
                prompt = TASK_PRE_PROMPT + "\n" + prompt

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
                    task_images=images,
                ):
                    messages.append(msg)
                    yield messages

                yield messages
        except Exception as e:
            yield messages
            raise gr.Error(f"Error: {str(e)}")

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
                    sanitize_html=False,
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

            # user input
            with Modal(visible=False, allow_user_close=False) as user_input_modal:
                question = gr.Markdown("")
                user_input = gr.Textbox("", lines=2, label="Type your answer:")
                submit_user_input = gr.Button("submit")

            def check_agent_question(s):
                try:
                    if "agent" in s:
                        if "user_input" in s["agent"].tools:
                            user_input_tool = s["agent"].tools["user_input"]
                            if isinstance(user_input_tool, UserInputTool):
                                if user_input_tool.is_waiting:
                                    return user_input_tool.question
                except:
                    pass
                return ""

            agent_question = gr.Textbox(
                check_agent_question,
                every=5,
                inputs=[session_state],
                visible=False,
                interactive=False,
            )

            def ask_user(question):
                if question:
                    return Modal(visible=True), gr.Markdown(question)
                else:
                    return Modal(visible=False), gr.Markdown("")

            agent_question.change(
                ask_user, inputs=[agent_question], outputs=[user_input_modal, question]
            )

            def answer_agent_question(s, content):
                try:
                    if "agent" in s:
                        if "user_input" in s["agent"].tools:
                            user_input_tool = s["agent"].tools["user_input"]
                            if isinstance(user_input_tool, UserInputTool):
                                if user_input_tool.is_waiting:
                                    user_input_tool.user_input = content
                                    user_input_tool.question = ""
                                    user_input_tool.is_waiting = False
                except:
                    pass

            submit_user_input.click(
                answer_agent_question,
                inputs=[session_state, user_input],
            ).then(
                lambda: (Modal(visible=False), "", ""),
                outputs=[user_input_modal, question, user_input],
            )

        demo.launch(debug=True, state_session_capacity=self.session_capacity, **kwargs)

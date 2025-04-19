TASK_PRE_PROMPT = """
Requirements:
- Conversations and code executions are consecutive from previous context, feel free to reuse existing variables and modules.
- Never just copy raw results from tools as final answer, you must think and make your answer.
- Never provide one-liner answer.
- Always provide a complete answer with final_answer tool even you have mention it before.
- You MUST use at most one tool at each step, and reflect extensively on the previous outcomes. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

"""


UPDATE_TOOL_PROMPT_PREFIX = """
Forget all previous tools. You are now working with a new set of tools, and you must only focus on these tools to assist with the task at hand.
Do **not** reference any previous tools, and only consider the current tools provided below.
Here are the tools currently available:"""

UPDATE_TOOL_PROMPT_TEMPLATE = (
    UPDATE_TOOL_PROMPT_PREFIX
    + '''
  {%- for tool in tools.values() %}
  def {{ tool.name }}({% for arg_name, arg_info in tool.inputs.items() %}{{ arg_name }}: {{ arg_info.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{tool.output_type}}:
      """{{ tool.description }}

      Args:
      {%- for arg_name, arg_info in tool.inputs.items() %}
          {{ arg_name }}: {{ arg_info.description }}
      {%- endfor %}
      """
  {% endfor %}
'''
)

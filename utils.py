import ast
import glob
import importlib.util
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from smolagents import Tool, DuckDuckGoSearchTool

from user_input_tool import UserInputTool


def verify_plugin_file(file_path):
    """
    Verifies that a Python file:
    1. Contains only class definitions and import statements.
    2. Defines a `tools` list containing only class references (not instances).
    3. Ensures each class's __init__ method can be created with no arguments or **kwargs.

    Returns: List of class names if valid, raises an error otherwise.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    class_names = set()
    tools_list = None
    classes = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):  # Collect class names
            class_names.add(node.name)
            classes.append(node)

        elif isinstance(node, ast.Assign):  # Look for the `tools = [...]` list
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "tools":
                    tools_list = node.value

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Allow import statements (safe)
            continue

        else:
            raise SyntaxError(f"Invalid top-level statement: {type(node).__name__}")

    # Ensure `tools` list exists
    if tools_list is None:
        raise ValueError(f"File {file_path} does not define a `tools` list.")

    # Validate `tools` contains only class names
    if not isinstance(tools_list, ast.List):
        raise TypeError(f"`tools` must be a list, but got {type(tools_list).__name__}")

    for element in tools_list.elts:
        if not isinstance(element, ast.Name) or element.id not in class_names:
            raise TypeError(
                f"`tools` must contain only class references (not instances). Invalid item: {element}"
            )

    # Ensure each class's __init__ method can be created with no arguments or **kwargs
    for cls in classes:
        init_method = next(
            (
                m
                for m in cls.body
                if isinstance(m, ast.FunctionDef) and m.name == "__init__"
            ),
            None,
        )
        if init_method:
            # Check the arguments of __init__
            args = [param for param in init_method.args.args if param.arg != "self"]
            if len(args) == 0:
                pass
            else:
                raise TypeError(
                    f"Class '{cls.name}' has an invalid __init__ method. It must have **kwargs only."
                )

    return list(class_names)


# Cache to store module timestamps and objects
_module_cache = {}
# Lock to protect _module_cache and _module_locks
_module_cache_lock = threading.Lock()
# Dictionary to hold per-module locks
_module_locks = {}


def import_variable(relative_path, variable_name):
    abs_path = Path(__file__).parent / relative_path

    # verify code first, prevent possible inf loop
    verify_plugin_file(relative_path)

    module_name = abs_path.stem
    last_modified = os.path.getmtime(abs_path)

    # First, acquire the global cache lock to check if the module is cached.
    with _module_cache_lock:
        if module_name in _module_cache:
            cached_time, cached_module = _module_cache[module_name]
            if cached_time == last_modified:
                return getattr(cached_module, variable_name, None)
        # Ensure there is a dedicated lock for this module.
        if module_name not in _module_locks:
            _module_locks[module_name] = threading.Lock()
        module_lock = _module_locks[module_name]

    # Use the per-module lock to ensure only one thread reloads the module.
    with module_lock:
        # Double-check: another thread may have updated the cache while waiting.
        with _module_cache_lock:
            if module_name in _module_cache:
                cached_time, cached_module = _module_cache[module_name]
                if cached_time == last_modified:
                    return getattr(cached_module, variable_name, None)

        # Load the module dynamically.
        spec = importlib.util.spec_from_file_location(module_name, str(abs_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {abs_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(
                f"Error loading module '{module_name}' from {abs_path}: {e}"
            )

        # Update the cache in a thread-safe manner.
        with _module_cache_lock:
            _module_cache[module_name] = (last_modified, module)

    if not hasattr(module, variable_name):
        raise AttributeError(f"Variable '{variable_name}' not found in '{module_name}'")

    return getattr(module, variable_name)


def load_plugin_files():
    return glob.glob("./plugins/**/*.py", recursive=True)


def load_plugin_tools(plugin: str):
    """Load tools from a single plugin file."""
    try:
        start_time = time.time()
        plugin_tools = import_variable(plugin, "tools")
        print("Loading {} took {} seconds".format(plugin, time.time() - start_time))
        valid_tools = []
        if isinstance(plugin_tools, Iterable):
            for tool in plugin_tools:
                if issubclass(tool, Tool):
                    valid_tools.append(tool)
        return valid_tools
    except Exception as e:
        print(f"Error loading plugin {plugin}: {e}")
        return []


def load_tools():
    available_tools: dict[str, Tool] = {
        DuckDuckGoSearchTool.name: DuckDuckGoSearchTool,
        UserInputTool.name: UserInputTool,
    }
    plugin_files = load_plugin_files()

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit each plugin loading job to the executor
        future_to_plugin = {
            executor.submit(load_plugin_tools, plugin): plugin
            for plugin in plugin_files
        }
        for future in as_completed(future_to_plugin):
            plugin = future_to_plugin[future]
            try:
                plugin_tools = future.result(timeout=30)
                for tool in plugin_tools:
                    if tool.name not in available_tools:
                        available_tools[tool.name] = tool
            except TimeoutError:
                print(f"Timeout loading plugin {plugin}, skipping.")
            except Exception as e:
                print(f"Error processing plugin {plugin}: {e}")

    return available_tools


if __name__ == "__main__":
    print(load_tools())

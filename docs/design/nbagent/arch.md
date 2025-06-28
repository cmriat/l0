# NBAgent

NBAgent is a notebook-based agent that executes tasks through a Jupyter notebook interface. It maintains state through notebook cells and provides tools for task execution.

## Architecture

```mermaid
classDiagram
    class NBAgent {
        -model: OpenAIServerModel
        -runtime: NBRuntime
        -tools: dict
        -state: NBState
        +run(task: str)
        +run_loop(task: str)
        -_setup_tools()
        -_setup_system_prompt()
    }
    
    class NBRuntime {
        -nb: NotebookNode
        +handle_new_cell(cell: NBCell)
        +register_tools(tools: list)
    }
    
    class NBState {
        -cells: list[NBCell]
        +to_messages()
    }
    
    class NBCell {
        -type: NBCellType
        -source: str
        -action_type: ActionType
        +to_messages()
    }

    NBAgent --> NBRuntime : uses
    NBAgent --> NBState : maintains
    NBState --> NBCell : contains
    NBRuntime --> NBCell : executes
```

## Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant NBAgent
    participant Model
    participant NBRuntime
    participant Tools

    User->>NBAgent: run(task)
    NBAgent->>NBAgent: _setup_task()
    NBAgent->>NBAgent: _setup_tools()
    NBAgent->>NBAgent: _setup_system_prompt()
    
    loop Until finished or max_steps
        NBAgent->>Model: Get next action
        Model-->>NBAgent: Response with think/code
        NBAgent->>NBRuntime: handle_new_cell()
        NBRuntime->>Tools: Execute tool calls
        Tools-->>NBRuntime: Tool results
        NBAgent->>NBAgent: Update state
    end

    NBAgent-->>User: Final answer
```

## Key Components

### 1. NBAgent
- Main agent class that orchestrates the execution
- Maintains state through notebook cells
- Handles tool registration and execution
- Manages conversation with the model

### 2. NBRuntime
- Manages Jupyter notebook execution
- Handles cell creation and execution
- Manages kernel lifecycle
- Registers tools in the notebook environment

### 3. NBState
- Represents current notebook state
- Contains list of cells
- Converts state to messages for model input

### 4. Tools System

```mermaid
classDiagram
    class NBTool {
        +name: str
        +prompt_descriptions: dict
        +nb_cell: NBCell
    }
    
    class ToolRegister {
        +register_func_factory()
        +register_class()
        +get_tool()
    }

    class BaseTools {
        +planning_tool
        +notepad_tool
        +submit_final_answer_tool
    }

    ToolRegister --> NBTool : creates
    BaseTools --> NBTool : implements
```

## Memory Management

```mermaid
flowchart TD
    A[User Input] --> B[NBAgent State]
    B --> C{Token Budget Check}
    C -->|Under Limit| D[Process Normally]
    C -->|Over Limit| E[Trim Memory]
    E --> F[Notepad]
    F --> G[Important Facts]
    F --> H[TODO List]
    F --> I[Draft]
    G --> J[Continue Processing]
    H --> J
    I --> J
```

## State Management

The agent maintains state through:
1. Notebook cells (NBCell)
2. Shared memory for:
   - TODO list
   - Important facts
   - Draft content
   - History tokens
   - Final answer
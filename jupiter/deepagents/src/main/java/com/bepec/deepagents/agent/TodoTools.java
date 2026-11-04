package com.bepec.deepagents.agent;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.agent.tool.Tool;

import java.util.ArrayList;
import java.util.List;

/** Planning tool (write_todos) as in the Python deepagents package. */
public class TodoTools {

    public static class Todo {
        @Description("what needs to be done")
        public String content;
        @Description("one of: pending, in_progress, completed")
        public String status;

        @Override
        public String toString() {
            return "[" + status + "] " + content;
        }
    }

    private final List<Todo> todos = new ArrayList<>();

    public List<Todo> todos() {
        return List.copyOf(todos);
    }

    @Tool(name = "write_todos", value = """
            Create or update the task plan. Always send the COMPLETE list of todos; \
            it replaces the previous list. Mark items in_progress / completed as you work.""")
    public String writeTodos(@P("the full todo list") List<Todo> items) {
        todos.clear();
        if (items != null) {
            todos.addAll(items);
        }
        StringBuilder sb = new StringBuilder("Updated todo list:\n");
        todos.forEach(t -> sb.append("- ").append(t).append('\n'));
        return sb.toString();
    }
}

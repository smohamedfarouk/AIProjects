package com.jupiter.guardrails;

import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.Tracing;
import com.jupiter.guardrails.common.SafetyGuard;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Executes the hierarchical graph as a state machine:
 * START -> guard -> supervisor -> (researcher | writer | coder) -> supervisor ... -> END
 */
public class AgentGraph {

    static final int RECURSION_LIMIT = 20;

    private final InputGuardrail guard;
    private final Supervisor supervisor;
    private final Map<String, Worker> workers = new LinkedHashMap<>();

    public AgentGraph(LlmClient llm) {
        SafetyGuard safety = new SafetyGuard(llm);
        addWorker(new Worker("researcher",
                "You are a researcher. Gather and summarize key facts for the task concisely.", llm, safety));
        addWorker(new Worker("writer",
                "You are a writer. Produce clear, polished prose for the task.", llm, safety));
        addWorker(new Worker("coder",
                "You are a coder. Write clean, correct code for the task with brief comments.", llm, safety));
        this.guard = new InputGuardrail(safety);
        this.supervisor = new Supervisor(llm, new ArrayList<>(workers.keySet()));
    }

    private void addWorker(Worker worker) {
        workers.put(worker.role(), worker);
    }

    public AgentState invoke(AgentState state) {
        int steps = 0;
        String node = "guard";
        while (node != null) {
            if (++steps > RECURSION_LIMIT) {
                throw new IllegalStateException("Recursion limit of " + RECURSION_LIMIT + " reached");
            }
            switch (node) {
                case "guard" -> {
                    guard.apply(state);
                    node = state.blocked() ? null : "supervisor";
                }
                case "supervisor" -> {
                    supervisor.apply(state);
                    node = AgentState.FINISH.equals(state.next()) ? null : state.next();
                }
                default -> {
                    // workers report back to supervisor (hierarchical loop)
                    workers.get(node).apply(state);
                    node = "supervisor";
                }
            }
        }
        return state;
    }

    public AgentState run(String task) {
        AgentState state = new AgentState(task);
        state.addMessage(UserMessage.from(task));
        AgentState fin = Tracing.trace("hierarchical_agent_run", "chain", span -> {
            span.setAttribute("input.value", task);
            AgentState result = invoke(state);
            Tracing.metadata(span, "blocked", result.blocked());
            span.setAttribute("output.value", LlmClient.content(result.messages().get(result.messages().size() - 1)));
            return result;
        });
        List<ChatMessage> messages = fin.messages();
        for (ChatMessage m : messages) {
            String type = m instanceof UserMessage ? "HUMAN" : "AI";
            System.out.println(type + ": " + LlmClient.content(m) + "\n");
        }
        return fin;
    }
}

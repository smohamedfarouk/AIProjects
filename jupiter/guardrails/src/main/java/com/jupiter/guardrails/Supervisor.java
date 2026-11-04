package com.jupiter.guardrails;

import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.Tracing;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/** Decides which worker acts next, or FINISH. */
public class Supervisor {

    static final String SUPERVISOR_SYSTEM =
            "You are a supervisor managing workers: %s. "
                    + "Given the task and conversation so far, decide who acts next. "
                    + "When the task is fully complete, respond with 'FINISH'. "
                    + "Reply with ONLY one word: a worker name or FINISH.";

    private final LlmClient llm;
    private final List<String> workers;

    public Supervisor(LlmClient llm, List<String> workers) {
        this.llm = llm;
        this.workers = workers;
    }

    public void apply(AgentState state) {
        Tracing.trace("supervisor", "chain", span -> {
            String history = state.messages().isEmpty()
                    ? "(none)"
                    : state.messages().stream().map(LlmClient::content).collect(Collectors.joining("\n"));
            span.setAttribute("input.value", history);
            String decision = llm.ask(
                    SUPERVISOR_SYSTEM.formatted(String.join(", ", workers)),
                    "Task: " + state.task() + "\n\nConversation so far:\n" + history
            ).strip().toLowerCase(Locale.ROOT);

            String nxt = AgentState.FINISH;
            for (String w : workers) {
                if (decision.contains(w)) {
                    nxt = w;
                    break;
                }
            }
            if (decision.contains("finish")) {
                nxt = AgentState.FINISH;
            }
            state.setNext(nxt);
            Tracing.metadata(span, "decision", nxt);
            span.setAttribute("output.value", nxt);
        });
    }
}

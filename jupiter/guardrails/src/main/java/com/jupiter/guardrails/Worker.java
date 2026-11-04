package com.jupiter.guardrails;

import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.Tracing;
import com.jupiter.guardrails.common.SafetyGuard;

import dev.langchain4j.data.message.AiMessage;

/** A role-specific worker whose output passes through the safety guardrail. */
public class Worker {

    private final String role;
    private final String system;
    private final LlmClient llm;
    private final SafetyGuard safety;

    public Worker(String role, String system, LlmClient llm, SafetyGuard safety) {
        this.role = role;
        this.system = system;
        this.llm = llm;
        this.safety = safety;
    }

    public String role() {
        return role;
    }

    public void apply(AgentState state) {
        Tracing.trace("worker_" + role, "chain", span -> {
            span.setAttribute("input.value", state.task());
            String raw = llm.ask(system, state.task());
            String result = outputGuardrail(raw);
            state.addMessage(AiMessage.from("[" + role + "]\n" + result));
            span.setAttribute("output.value", result);
        });
    }

    // output guardrail per worker
    private String outputGuardrail(String raw) {
        return Tracing.trace("output_guardrail", "chain", span -> {
            span.setAttribute("input.value", raw);
            SafetyGuard.GuardResult check = safety.check(raw);
            Tracing.metadata(span, "guard.stage", "output");
            Tracing.metadata(span, "guard.worker", role);
            Tracing.metadata(span, "guard.blocked", !check.safe());
            Tracing.metadata(span, "guard.keyword_hit", check.keywordHit());
            Tracing.metadata(span, "guard.verdict", check.classifierVerdict());
            String result = check.safe() ? raw : "[" + role + "] Output blocked by guardrail.";
            span.setAttribute("output.value", check.safe() ? "ALLOWED" : "BLOCKED");
            return result;
        });
    }
}

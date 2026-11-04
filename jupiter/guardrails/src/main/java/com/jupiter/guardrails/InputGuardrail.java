package com.jupiter.guardrails;

import com.jupiter.guardrails.common.Tracing;
import com.jupiter.guardrails.common.SafetyGuard;

import dev.langchain4j.data.message.AiMessage;

/** Entry node: blocks the whole run if the incoming task is unsafe. */
public class InputGuardrail {

    private final SafetyGuard safety;

    public InputGuardrail(SafetyGuard safety) {
        this.safety = safety;
    }

    public void apply(AgentState state) {
        Tracing.trace("input_guardrail", "chain", span -> {
            span.setAttribute("input.value", state.task());
            SafetyGuard.GuardResult result = safety.check(state.task());
            if (!result.safe()) {
                state.setBlocked(true);
                state.addMessage(AiMessage.from("Request blocked by guardrail."));
                state.setNext(AgentState.FINISH);
            } else {
                state.setBlocked(false);
            }
            Tracing.metadata(span, "guard.stage", "input");
            Tracing.metadata(span, "guard.blocked", state.blocked());
            Tracing.metadata(span, "guard.keyword_hit", result.keywordHit());
            Tracing.metadata(span, "guard.verdict", result.classifierVerdict());
            span.setAttribute("output.value", state.blocked() ? "BLOCKED" : "ALLOWED");
        });
    }
}

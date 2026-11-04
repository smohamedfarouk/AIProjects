package com.jupiter.guardrails.common.eval;

import java.util.Map;

/** Scores one example run, like a Python evaluator function passed to langsmith evaluate(). */
@FunctionalInterface
public interface Evaluator {

    Result evaluate(Case c);

    /** What an evaluator can see: the example inputs, the target's outputs and the reference outputs. */
    record Case(Map<String, Object> inputs, Map<String, Object> outputs, Map<String, Object> referenceOutputs) {

        public String input(String key) {
            return str(inputs, key);
        }

        public String output(String key) {
            return str(outputs, key);
        }

        public String reference(String key) {
            return str(referenceOutputs, key);
        }

        public Object rawOutput(String key) {
            return outputs == null ? null : outputs.get(key);
        }

        private static String str(Map<String, Object> map, String key) {
            Object v = map == null ? null : map.get(key);
            return v == null ? "" : v.toString();
        }
    }

    /** Feedback for one key; score may be null (no score), like {"key": ..., "score": None}. */
    record Result(String key, Double score, String comment) {

        public static Result of(String key, double score) {
            return new Result(key, score, null);
        }

        public static Result of(String key, double score, String comment) {
            return new Result(key, score, comment);
        }
    }
}

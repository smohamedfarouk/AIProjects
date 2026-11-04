package com.jupiter.guardrails.other;

import com.jupiter.guardrails.common.Config;
import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.SafetyGuard;
import com.jupiter.guardrails.common.Tracing;
import com.jupiter.guardrails.common.eval.Evaluator;
import com.jupiter.guardrails.common.eval.ExperimentRunner;
import com.jupiter.guardrails.common.eval.LangSmithClient;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Java port of guardrails-other-python/otherguardrails.py:
 * a guarded chatbot (keyword + length input guardrail, LLM moderation on input and output)
 * evaluated in LangSmith with a custom correctness judge and a QA grader.
 */
public class OtherGuardrailsMain {

    static final String MODEL_NAME = "gpt-4o-mini";
    static final String PROJECT = "simple-chatbot-eval2";
    static final String DATASET_NAME = "chatbot-eval-dataset2";

    static final List<String> BLOCKED_TOPICS = List.of("weapons", "self-harm", "illegal drugs", "hacking", "violence");
    static final int MAX_INPUT_LENGTH = 2000;

    static final String CHAT_SYSTEM = "You are a helpful, polite assistant. Keep answers concise.";

    static final String CORRECTNESS_SYSTEM = "Return only a number 1 (correct) or 0 (incorrect). "
            + "Compare the prediction to the reference for factual correctness.";

    // Same grading prompt as LangChain's built-in "qa" string evaluator.
    static final String QA_PROMPT = """
            You are a teacher grading a quiz.
            You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

            Example Format:
            QUESTION: question here
            STUDENT ANSWER: student's answer here
            TRUE ANSWER: true answer here
            GRADE: CORRECT or INCORRECT here

            Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

            QUESTION: %s
            STUDENT ANSWER: %s
            TRUE ANSWER: %s
            GRADE:""";

    static final List<LangSmithClient.Example> EXAMPLES = List.of(
            LangSmithClient.Example.of(Map.of("question", "Who won IPL 2026?"),
                    Map.of("answer", "RCB")),
            LangSmithClient.Example.of(Map.of("question", "Explain photosynthesis in one line."),
                    Map.of("answer", "Plants convert sunlight, water, and CO2 into glucose and oxygen.")),
            LangSmithClient.Example.of(Map.of("question", "What is 12 * 8?"),
                    Map.of("answer", "96"))
    );

    /** Result of the rule-based input guardrail: ok, or the message to return instead. */
    record InputCheck(boolean ok, String message) {
    }

    private final LlmClient llm;
    private final SafetyGuard moderator;

    public OtherGuardrailsMain(LlmClient llm) {
        this.llm = llm;
        this.moderator = new SafetyGuard(llm);
    }

    static InputCheck inputGuardrail(String userInput) {
        String text = userInput.toLowerCase(Locale.ROOT);
        for (String topic : BLOCKED_TOPICS) {
            if (text.contains(topic)) {
                return new InputCheck(false, "Request blocked: relates to restricted topic '" + topic + "'.");
            }
        }
        if (userInput.length() > MAX_INPUT_LENGTH) {
            return new InputCheck(false, "Request blocked: input too long.");
        }
        return new InputCheck(true, "");
    }

    // LLM-based moderation guardrail
    boolean llmGuardrail(String text) {
        return moderator.classify(text).startsWith("SAFE");
    }

    String outputGuardrail(String response) {
        if (!llmGuardrail(response)) {
            return "I'm sorry, I can't share that response.";
        }
        return response;
    }

    public String chatbot(String question) {
        return Tracing.trace("guarded_chatbot", "chain", span -> {
            span.setAttribute("input.value", question);
            String answer;
            InputCheck check = inputGuardrail(question);
            if (!check.ok()) {
                answer = check.message();
            } else if (!llmGuardrail(question)) {
                answer = "I'm sorry, I can't help with that request.";
            } else {
                answer = outputGuardrail(llm.ask(CHAT_SYSTEM, question));
            }
            span.setAttribute("output.value", answer);
            return answer;
        });
    }

    Evaluator correctness() {
        return c -> {
            String score = llm.ask(CORRECTNESS_SYSTEM,
                    "Prediction: " + c.output("answer") + "\nReference: " + c.reference("answer")).strip();
            return Evaluator.Result.of("correctness", score.startsWith("1") ? 1 : 0);
        };
    }

    // Built-in style evaluator (LangChainStringEvaluator("qa")); logged as "qa" so it doesn't clash with correctness.
    Evaluator qaEvaluator() {
        return c -> {
            String grade = llm.ask(QA_PROMPT.formatted(c.input("question"), c.output("answer"), c.reference("answer")))
                    .strip().toUpperCase(Locale.ROOT).replaceFirst("^GRADE:\\s*", "");
            return Evaluator.Result.of("qa", grade.startsWith("CORRECT") ? 1 : 0, grade);
        };
    }

    public static void main(String[] args) {
        Tracing.init(PROJECT);
        try {
            LlmClient llm = new LlmClient(OpenAiChatModel.builder()
                    .apiKey(Config.OPENAI_API_KEY)
                    .modelName(MODEL_NAME)
                    .temperature(0.0)
                    .build(), MODEL_NAME);
            OtherGuardrailsMain bot = new OtherGuardrailsMain(llm);

            LangSmithClient client = LangSmithClient.fromConfig();
            client.ensureDataset(DATASET_NAME, EXAMPLES);

            ExperimentRunner.evaluate(client,
                    inputs -> Map.of("answer", bot.chatbot(inputs.get("question").toString())),
                    DATASET_NAME,
                    List.of(bot.correctness(), bot.qaEvaluator()),
                    "guarded-chatbot");
        } finally {
            Tracing.shutdown();
        }
    }
}

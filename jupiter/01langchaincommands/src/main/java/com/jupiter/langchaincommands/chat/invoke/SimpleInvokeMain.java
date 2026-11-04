package com.jupiter.langchaincommands.chat.invoke;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;

/**
 * More LangChain commands, lesson 1 - the simplest possible LLM call (morelangchaincommands.ipynb cells 0-2).
 * <b>Entry point.</b>
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@invoke
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>What it shows</h2>
 * {@code llm.invoke("What is AI in 1 Line")} sends one user message and returns an {@code AIMessage}. Besides the
 * text ({@code content}), the reply carries metadata that the notebook printed too:
 * <ul>
 *   <li><b>model name</b> - the exact model version that answered, e.g. gpt-4o-mini-2024-07-18;</li>
 *   <li><b>token usage</b> - input (prompt) and output (completion) tokens; OpenAI bills by these;</li>
 *   <li><b>finish reason</b> - why the model stopped: STOP (done), LENGTH (hit the token limit), TOOL_EXECUTION
 *       (wants a tool);</li>
 *   <li><b>id</b> - OpenAI's id for this response, useful when tracing a request.</li>
 * </ul>
 *
 * <h2>Notebook result</h2>
 * <pre>
 * "AI, or artificial intelligence, refers to the simulation of human intelligence processes by machines,
 *  particularly computer systems."   - 14 prompt tokens, 21 completion tokens, 35 total
 * </pre>
 */
public final class SimpleInvokeMain {

    static final String QUESTION = "What is AI in 1 Line";

    private SimpleInvokeMain() {
    }

    /**
     * Entry point - {@code ChatOpenAI(model="gpt-4o-mini").invoke("What is AI in 1 Line")}.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Create gpt-4o-mini and send the question as one user message.</li>
     *   <li>Print the answer text, then the response metadata (model, tokens, finish reason, id).</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("llm.invoke(\"" + QUESTION + "\")");
        ChatModel llm = Models.openAi("gpt-4o-mini");
        ChatResponse response = llm.chat(UserMessage.from(QUESTION));

        System.out.println(response.aiMessage().text());
        System.out.println("\nmodel:         " + response.modelName());
        System.out.println("token usage:   " + response.tokenUsage());
        System.out.println("finish reason: " + response.finishReason());
        System.out.println("id:            " + response.id());
    }
}

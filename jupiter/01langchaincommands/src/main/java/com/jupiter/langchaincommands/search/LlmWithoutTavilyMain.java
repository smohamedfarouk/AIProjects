package com.jupiter.langchaincommands.search;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;

/**
 * Requirement 1, part A - the LLM <b>without</b> Tavily internet search (notebook cells 0-2). <b>Entry point.</b>
 * <p>
 * Asks gpt-5.4 directly for today's AI news, with no tools. Run this first, then {@link DeepAgentWithTavilyMain}
 * with the same question, and compare the two answers: that difference is the reason deep agents with a search tool
 * exist.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@without-tavily
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs only OPENAI_API_KEY (environment or .env) - no Tavily key.
 * Makes one paid OpenAI call.
 *
 * <h2>The problem: an LLM only knows its training data</h2>
 * The question is "Latest News in AI Space as of Sep 16th 2026". The model has no internet access and no tools, so
 * it can only answer from what it learned during training, which ends at its training cut-off date. It cannot know
 * what happened on a specific recent day. In the notebook it gave general 2026 trends and said itself that it cannot
 * browse. Fresh facts need a tool that fetches them - see {@link DeepAgentWithTavilyMain}.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does                                   | Result                                                       |
 * |------|------------------------------------------------|--------------------------------------------------------------|
 * | 2    | Asks gpt-5.4 directly for AI news on           | It answered with general 2026 trends and said itself that it |
 * |      | Sep 16, 2026                                   | can't browse.                                                |
 * </pre>
 */
public final class LlmWithoutTavilyMain {

    /** Same question as {@link DeepAgentWithTavilyMain} asks, so the two answers can be compared. */
    static final String NEWS_QUESTION = DeepAgentWithTavilyMain.NEWS_QUESTION;

    private LlmWithoutTavilyMain() {
    }

    /**
     * Entry point - the Java version of {@code llm = ChatOpenAI(model="gpt-5.4")} and
     * {@code llm.invoke("Latest News in AI Space as of Sep 16th 2026")}.
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Create gpt-5.4 with no tools.</li>
     *   <li>Send the question as a single user message.</li>
     *   <li>Print the answer, plus why the model stopped (finish reason, normally STOP) and how many tokens it used -
     *       the extra fields of the response that the notebook also showed.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        //Not up to date informaiton
        Models.section("llm.invoke(...) with gpt-5.4 - no internet search");
        ChatModel gpt54 = Models.openAi("gpt-5.4");
        ChatResponse response = gpt54.chat(UserMessage.from(NEWS_QUESTION));
        System.out.println(response.aiMessage().text());
        System.out.println("\n(finish reason: " + response.finishReason() + ", " + response.tokenUsage() + ")");
    }
}

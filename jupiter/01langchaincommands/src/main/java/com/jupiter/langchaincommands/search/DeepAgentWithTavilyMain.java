package com.jupiter.langchaincommands.search;

import com.jupiter.langchaincommands.common.Config;
import com.jupiter.langchaincommands.common.Models;

/**
 * Requirement 1, part B - a deep agent <b>with</b> Tavily internet search (notebook cells 3-9). <b>Entry point.</b>
 * <p>
 * Ask a question to the LLM model; if the answer is not there, it needs to search the internet and get the response.
 * Deep agents for building a compound AI system.
 * <p>
 * Asks the same question as {@link LlmWithoutTavilyMain} (which answers from training data only), but this time the
 * model can search the web, so it can answer with real, dated news.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@with-tavily
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY and TAVILY_API_KEY (environment or .env).
 * Makes several paid OpenAI calls (one per agent step) and Tavily searches.
 *
 * <h2>The fix: a deep agent with a search tool</h2>
 * A {@link DeepAgent} gets the question and can call {@link TavilySearch} (Tavily is a search API built for LLMs: it
 * returns clean text snippets with titles and URLs instead of raw web pages).
 * <pre>
 *   question
 *     -> agent plans:          write_todos ["[pending] search news", "[pending] save notes", "[pending] report"]
 *     -> agent researches:     internet_search("AI news September 16 2026", 5)   (repeated as needed)
 *     -> agent saves findings: write_file("/tmp/ai_news_sep16_2026_report.md", ...)
 *     -> agent answers:        final report with real, dated headlines
 * </pre>
 * The model decides every step itself - which tool to call, with what arguments, how often, and when it is done.
 * Our code only defines the tools and runs the loop (LangChain4j's AiServices does the call-tool-then-continue loop).
 *
 * <h2>What makes it a "deep" agent</h2>
 * A plain tool-calling agent just loops: think -> call tool -> read result -> repeat. deepagents adds three things
 * that help with long, open-ended tasks, and {@link DeepAgent} reproduces them:
 * <ul>
 *   <li><b>Planning</b> - a write_todos tool, so the model writes and updates its own to-do list.</li>
 *   <li><b>A virtual file system</b> - ls / read_file / write_file / edit_file kept in memory, so the model can save
 *       notes and drafts instead of holding everything in the conversation.</li>
 *   <li><b>A detailed system prompt</b> - the user's instructions plus built-in guidance on using those tools.</li>
 * </ul>
 * (Python deepagents can also start helper sub-agents; this small version does not need them.)
 *
 * <h2>When to use what</h2>
 * A simple rule: if you can draw the workflow as a flowchart ahead of time, use LangGraph. If you can only describe
 * the goal and want the agent to work out the steps, use Deep Agents. Not much control of the flow -> deep agents;
 * a lot of control -> LangGraph. See the table in {@link DeepAgent}.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does                                   | Result                                                       |
 * |------|------------------------------------------------|--------------------------------------------------------------|
 * | 8-9  | Deep agent with Tavily web search, same        | Gave real dated headlines: EU push to slow frontier AI,      |
 * |      | question                                       | OpenAI funding talks at about $1.2T, Anthropic's wet-lab     |
 * |      |                                                | biology work. It also wrote a report file,                   |
 * |      |                                                | /tmp/ai_news_sep16_2026_report.md, into its virtual files.   |
 * </pre>
 *
 * <h2>Classes</h2>
 * <pre>
 * | Class                   | What it covers                                                                    |
 * |-------------------------|-----------------------------------------------------------------------------------|
 * | DeepAgentWithTavilyMain | Entry point: the deep agent with internet search (this class)                     |
 * | LlmWithoutTavilyMain    | Entry point: the same question to the plain LLM, for comparison                   |
 * | DeepAgent               | Replaces create_deep_agent: to-do list, in-memory files and internet search tools |
 * | TavilySearch            | Replaces TavilyClient(api_key=...).search(query=..., max_results=...)             |
 * </pre>
 */
public final class DeepAgentWithTavilyMain {

    static final String NEWS_QUESTION = "Latest News in AI Space as of Sep 16th 2026";

    static final String RESEARCHER_PROMPT = "You are a web agent that can search the internet for information."
            + "You are an expert Researcher, Always (1) Use Write-Todos to Plan"
            + "(2) Use Internet Search Tool to gather Information, (3) Save findings to file"
            + "(4) write a final report";

    private DeepAgentWithTavilyMain() {
    }

    /**
     * Entry point - the Java version of
     * {@code create_deep_agent(tools=[internet_search], system_prompt=..., model=llm)} and
     * {@code web_agent.invoke({"messages": [{"role": "user", "content": ...}]})}.
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Create the Tavily client - fails here with a clear message if TAVILY_API_KEY is missing.</li>
     *   <li>Build the {@link DeepAgent} on gpt-5.4 with strict tool arguments (always valid JSON) and the researcher
     *       system prompt: plan with to-dos, search the internet, save findings to a file, write a final report.</li>
     *   <li>Ask the news question. While the agent works, each tool call is printed as {@code [tool] name: ...}.</li>
     *   <li>Print the final report and the files the agent saved in its virtual file system.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        //Up to date information using tavily internet search
        Models.section("Deep agent with Tavily internet search");
        DeepAgent webAgent = new DeepAgent(Models.openAi("gpt-5.4", true),
                new TavilySearch(Config.TAVILY_API_KEY), RESEARCHER_PROMPT);
        System.out.println(webAgent.invoke(NEWS_QUESTION));
        webAgent.files().forEach((path, content) ->
                System.out.println("\n[saved file] " + path + " (" + content.length() + " chars)"));
    }
}

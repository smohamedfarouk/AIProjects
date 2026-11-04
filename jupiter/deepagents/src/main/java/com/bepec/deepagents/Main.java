package com.bepec.deepagents;

import com.bepec.deepagents.agent.AgentResult;
import com.bepec.deepagents.agent.DeepAgent;
import com.bepec.deepagents.agent.MessagePrinter;
import com.bepec.deepagents.agent.SubAgent;
import com.bepec.deepagents.tools.ClassifyIntentTool;
import com.bepec.deepagents.tools.ExpandKeywordsTool;
import com.bepec.deepagents.tools.InternetSearchTool;
import dev.langchain4j.model.chat.ChatModel;

import java.util.List;

/**
 * Java port of {@code deepagents.ipynb}.
 * <p>
 * Usage: {@code mvn -q compile exec:java -Dexec.args="<demo>"} where demo is one of
 * {@code llm | research | stream | ads | all} (default: {@code research}).
 * <p>
 * Classes in this project:
 * <pre>
 * ┌─────┬────────────────────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
 * │ #   │ Class                  │ Package              │ What it does                                                                                │ Why it exists                                                           │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 1   │ Main                   │ (root)               │ Entry point that runs one demo: llm, research, stream, ads or all                           │ Reproduces the notebook cells as runnable Java                          │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 2   │ Config                 │ (root)               │ Loads .env, reads API keys and builds the OpenAI ChatModel (gpt-5.4-mini, temperature 0)    │ Java version of load_dotenv() + ChatOpenAI(); keeps key and model setup │
 * │     │                        │                      │                                                                                             │ in one place                                                            │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 3   │ DeepAgent              │ ...agent             │ Agent loop: calls the LLM, runs tool calls and repeats until a final answer (max 50 steps). │ Java version of create_deep_agent() - the core of the project           │
 * │     │                        │                      │ Adds the task tool when subagents are set                                                   │                                                                         │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 4   │ DeepAgent.Builder      │ ...agent             │ Fluent setup: .model(), .systemPrompt(), .tools(), .subagents()                             │ Readable agent configuration, like Python keyword arguments             │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 5   │ SubAgent               │ ...agent             │ Record: name, description, systemPrompt, tools                                              │ Java version of the notebook's subagent dicts; the supervisor delegates │
 * │     │                        │                      │                                                                                             │ to them via the task tool                                               │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 6   │ AgentResult            │ ...agent             │ Record holding the full message list and the files snapshot. lastText() returns the final   │ Java version of the Python output {"messages": [...], "files": {...}}   │
 * │     │                        │                      │ answer                                                                                      │ and out["messages"][-1].content                                         │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 7   │ TodoTools              │ ...agent             │ write_todos tool that stores the agent's plan (pending / in_progress / completed)           │ Planning is built into every deep agent, so the model can break work    │
 * │     │                        │                      │                                                                                             │ into steps                                                              │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 8   │ TodoTools.Todo         │ ...agent             │ One todo item: content + status                                                             │ Typed structure so the LLM sends todos as JSON objects                  │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 9   │ VirtualFileSystemTools │ ...agent             │ In-memory ls, read_file, write_file, edit_file                                              │ The supervisor and subagents share files through it (seeds.md →         │
 * │     │                        │                      │                                                                                             │ expanded.md → final.md) without touching disk                           │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 10  │ MessagePrinter         │ ...agent             │ Formats System/Human/AI/Tool messages with headers                                          │ Java version of LangChain's message.pretty_print(), used by the stream  │
 * │     │                        │                      │                                                                                             │ demo                                                                    │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 11  │ InternetSearchTool     │ ...tools             │ internet_search tool that calls the Tavily REST API                                         │ Gives agents live web data (news, jobs, keyword research)               │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 12  │ ExpandKeywordsTool     │ ...tools             │ expand_keywords tool that adds modifiers (course, training, online, 2026…) to a seed        │ Used by expansion-agent to generate long-tail Google Ads keywords       │
 * │     │                        │                      │ keyword                                                                                     │                                                                         │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 13  │ ClassifyIntentTool     │ ...tools             │ classify_intent tool that labels a keyword informational, commercial or navigational, with  │ Used by classify-agent to sort keywords and spot negative ones          │
 * │     │                        │                      │ low/high/medium priority                                                                    │                                                                         │
 * ├─────┼────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
 * │ 14  │ ToolsTest              │ test                 │ Unit tests for the two keyword tools, plus a scripted fake-LLM test of supervisor →         │ Checks the logic without calling OpenAI or Tavily, so it costs nothing  │
 * │     │                        │                      │ subagent with shared files                                                                  │ to run                                                                  │
 * └─────┴────────────────────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘
 * </pre>
 */
public class Main {

    public static void main(String[] args) {
        String demo = args.length > 0 ? args[0] : "research";
        ChatModel llm = Config.llm();

        switch (demo) {
            case "llm" -> plainLlm(llm);
            case "research" -> researchAgent(llm);
            case "stream" -> streamResearchAgent(llm);
            case "ads" -> googleAdsSupervisor(llm);
            case "all" -> {
                plainLlm(llm);
                researchAgent(llm);
                streamResearchAgent(llm);
                googleAdsSupervisor(llm);
            }
            default -> {
                System.err.println("Unknown demo '" + demo + "'. Use one of: llm, research, stream, ads, all");
                System.exit(1);
            }
        }
    }

    // Cells 3-4 and 13: llm.invoke(...)
    static void plainLlm(ChatModel llm) {
        System.out.println(llm.chat("what is deep agents"));
        System.out.println(llm.chat("LAtest news from Apple as of June9th 2026"));
    }

    // Cells 6-10: Tavily + research deep agent
    static DeepAgent buildResearchAgent(ChatModel llm) {
        InternetSearchTool internetSearch = new InternetSearchTool(Config.env("TAVILY_API_KEY"));
        return DeepAgent.builder()
                .tools(internetSearch)
                .systemPrompt("You are an expert researcher. Always: (1) Use write_todos to plan,"
                        + "(2) use internet_search to gather facts, (3) save findings to file,"
                        + "(4) write a clear final report")
                .model(llm)
                .build();
    }

    // Cells 11-12: research_agent.invoke(...); print(out["messages"][-1].content)
    static void researchAgent(ChatModel llm) {
        DeepAgent researchAgent = buildResearchAgent(llm);
        AgentResult out = researchAgent.invoke("Research the latest news from apple as of June 9th 2026");
        System.out.println(out.lastText());
    }

    // Cell 14: research_agent.stream(..., stream_mode="values") -> pretty_print each message
    static void streamResearchAgent(ChatModel llm) {
        DeepAgent researchAgent = buildResearchAgent(llm);
        researchAgent.stream("Latest Job Posting on AI Engineer as of June 9th 2026", MessagePrinter::prettyPrint);
    }

    // Cells 15-20: Google Ads supervisor with three subagents
    static void googleAdsSupervisor(ChatModel llm) {
        InternetSearchTool internetSearch = new InternetSearchTool(Config.env("TAVILY_API_KEY"));
        ExpandKeywordsTool expandKeywords = new ExpandKeywordsTool();
        ClassifyIntentTool classifyIntent = new ClassifyIntentTool();

        SubAgent researchSub = new SubAgent(
                "research-agent",
                "finds seed keywords by researching the topic. Use FIRST",
                "You are a keyword research specialist. Use internet_seach to find"
                        + "10-15 high value SEED Keywords. Save them to 'seeds.md'.",
                List.of(internetSearch));

        SubAgent expansionSub = new SubAgent(
                "expansion-agent",
                "expands seed keywords into long-tail variations. Use SECOND",
                "Read 'seeds.md' for each seed call expand_keywords. deduplicate"
                        + "and save everything to 'expanded.md'.",
                List.of(expandKeywords));

        SubAgent classifySub = new SubAgent(
                "classify-agent",
                "classifies keywords by intent and ad priority and flag negatives. Use THIRD",
                "Read 'expanded.md' for each keyword call classify_intent. deduplicate"
                        + "Group into High Priority, Commercial, informational and negative keywords and save to final.md",
                List.of(classifyIntent));

        DeepAgent supervisor = DeepAgent.builder()
                .tools(internetSearch, expandKeywords, classifyIntent)
                .systemPrompt("You are the Supervisor of a Google Ads Team in BEPEC. Use write_todos to plan"
                        + "then delegate in order: research_agent, expansion_agent, classify_agent"
                        + "never do the work yourself. finally read final.md and summarise the top 10"
                        + "commerical keywords plus a negative keyword list")
                .model(llm)
                .subagents(researchSub, expansionSub, classifySub)
                .build();

        AgentResult out = supervisor.invoke(
                "Find google ads keywords for AI Training for Working Professionals from India, UK, USA");
        System.out.println(out.lastText());
    }
}

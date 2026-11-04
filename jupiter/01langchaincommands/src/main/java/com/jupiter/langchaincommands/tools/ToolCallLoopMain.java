package com.jupiter.langchaincommands.tools;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;

import java.util.ArrayList;
import java.util.List;

/**
 * More LangChain commands, lesson 4b - tool calling end to end: run the tools and send the results back
 * (morelangchaincommands.ipynb cells 24-25). <b>Entry point.</b>
 * <p>
 * Builds on {@link ToolCallRequestMain}, which only showed what the model asks for.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@tool-loop
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). Two paid OpenAI calls.
 *
 * <h2>The tool-calling loop, written out by hand</h2>
 * <pre>
 *   messages = [Human: question]
 *   1. model(messages)            -> AI: tool calls get_course_fee("Data Science"), get_batch_dates("Data Science")
 *   2. for each tool call: run it -> Tool: "INR 75000" (id call_1), Tool: "1st Oct 2026" (id call_2)
 *   3. model(messages)            -> AI: "The fee is INR 75,000 and the next batch starts on 1st October 2026."
 * </pre>
 * Each tool result is linked to its call by the call's id, so the model knows which answer belongs to which question.
 * The model must see the whole conversation again in step 3 - it remembers nothing between calls.
 * <p>
 * Doing it by hand shows what agent frameworks do for you: LangChain4j's AiServices (used by the deep agent in the
 * {@code search} package) runs exactly this loop automatically, repeating it until the model stops asking for tools.
 *
 * <h2>Notebook result</h2>
 * "The fee for the Data Science course is INR 75,000. The next batch is starting on 1st October 2026."
 */
public final class ToolCallLoopMain {

    private ToolCallLoopMain() {
    }

    /**
     * One full round: ask -> run the requested tools -> ask again with the results.
     *
     * @param llm      the chat model (OpenAI in production, a scripted fake in tests)
     * @param tools    the tool implementations
     * @param question the user's question
     * @return every message of the conversation; the last one is the model's final answer
     */
    static List<ChatMessage> run(ChatModel llm, CourseTools tools, String question) {
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(UserMessage.from(question));                                   // messages = [HumanMessage(question)]

        AiMessage response = ask(llm, messages);                                    // response = llm_tools.invoke(messages)
        messages.add(response);                                                     // messages.append(response)

        if (response.hasToolExecutionRequests()) {
            for (ToolExecutionRequest call : response.toolExecutionRequests()) {    // for tool_call in response.tool_calls
                String result = tools.execute(call);                                //   tool.invoke(tool_call["args"])
                messages.add(ToolExecutionResultMessage.from(call, result));        //   ToolMessage(content, tool_call_id)
            }
            messages.add(ask(llm, messages));                                       // final_response = llm_tools.invoke(messages)
        }
        return messages;
    }

    private static AiMessage ask(ChatModel llm, List<ChatMessage> messages) {
        return llm.chat(ChatRequest.builder()
                .messages(messages)
                .toolSpecifications(CourseTools.specifications())
                .build()).aiMessage();
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Run the loop ({@link #run}) on gpt-4o-mini, printing each tool call and its result as it happens.</li>
     *   <li>Print the final answer ({@code final_response.content}).</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("manual tool-calling loop");
        List<ChatMessage> messages = run(Models.openAi("gpt-4o-mini"), new CourseTools(), ToolCallRequestMain.QUESTION);
        for (ChatMessage m : messages) {
            if (m instanceof ToolExecutionResultMessage t) {
                System.out.println("[tool] " + t.toolName() + " -> " + t.text());
            }
        }
        System.out.println("\n" + ((AiMessage) messages.getLast()).text());
    }
}

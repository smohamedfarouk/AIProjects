package com.jupiter.langchaincommands.tools;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;

import java.util.List;

/**
 * More LangChain commands, lesson 4a - tool calling: what the model <b>asks for</b>
 * (morelangchaincommands.ipynb cells 18-23). <b>Entry point.</b>
 * <p>
 * Next step: {@link ToolCallLoopMain} actually runs the tools and sends the results back.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@tool-calls
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>What it shows</h2>
 * The model gets the question plus the descriptions of two tools ({@link CourseTools}). It cannot know BEPEC's fees
 * or dates, so instead of answering it replies with <b>tool calls</b>: which tool, with which arguments, and an id
 * for each call. Nothing has been executed yet - this is only the model's request.
 * <pre>
 *   "What is the fee for Data Science course and when is the next batch starting?"
 *     -> tool call get_course_fee(course_name="Data Science")   id call_Ttb...
 *     -> tool call get_batch_dates(course_name="Data Science")  id call_KFU...
 * </pre>
 * The model asked for both tools in one reply because the question needs both answers.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * [{'name': 'get_course_fee',  'args': {'course_name': 'Data Science'}, 'id': 'call_TtbTx...'},
 *  {'name': 'get_batch_dates', 'args': {'course_name': 'Data Science'}, 'id': 'call_KFUrF...'}]
 * </pre>
 */
public final class ToolCallRequestMain {

    static final String QUESTION = "What is the fee for Data Science course and when is the next batch starting?";

    private ToolCallRequestMain() {
    }

    /**
     * {@code llm.bind_tools([...]).invoke(question).tool_calls}: one model call with the tool descriptions attached.
     *
     * @return the tool calls the model asked for (empty if it answered in text instead)
     */
    static List<ToolExecutionRequest> requestedToolCalls(ChatModel llm, String question) {
        AiMessage reply = llm.chat(ChatRequest.builder()
                .messages(UserMessage.from(question))
                .toolSpecifications(CourseTools.specifications())
                .build()).aiMessage();
        return reply.hasToolExecutionRequests() ? reply.toolExecutionRequests() : List.of();
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Send the question to gpt-4o-mini together with the two tool specifications.</li>
     *   <li>Print each requested tool call: name, arguments (JSON) and id - like {@code response.tool_calls}.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("llm.bind_tools([get_course_fee, get_batch_dates]).invoke(question).tool_calls");
        List<ToolExecutionRequest> calls = requestedToolCalls(Models.openAi("gpt-4o-mini"), QUESTION);
        if (calls.isEmpty()) {
            System.out.println("(the model answered directly, without asking for a tool)");
        }
        calls.forEach(c -> System.out.println("name: " + c.name() + "  args: " + c.arguments() + "  id: " + c.id()));
    }
}

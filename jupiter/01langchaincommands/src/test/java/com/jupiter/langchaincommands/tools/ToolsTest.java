package com.jupiter.langchaincommands.tools;

import com.jupiter.langchaincommands.testutil.Fakes.ScriptedChatModel;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests of the tool-calling lessons (no OpenAI calls). */
class ToolsTest {

    static ToolExecutionRequest call(String id, String name, String json) {
        return ToolExecutionRequest.builder().id(id).name(name).arguments(json).build();
    }

    @Test
    void toolsLookUpFeesAndDatesIgnoringCase() {
        CourseTools tools = new CourseTools();
        assertEquals("INR 75000", tools.getCourseFee("Data Science"));
        assertEquals("1st March 2027", tools.getBatchDates("AI COURSE"));
        assertEquals("Course not found", tools.getCourseFee("Cooking"));
    }

    @Test
    void specificationsUseTheNotebookToolNames() {
        assertEquals(List.of("get_batch_dates", "get_course_fee"),
                CourseTools.specifications().stream().map(ToolSpecification::name).sorted().toList());
    }

    @Test
    void executeDispatchesByNameAndAcceptsBothArgumentStyles() {
        CourseTools tools = new CourseTools();
        assertEquals("INR 85000", tools.execute(call("1", "get_course_fee", "{\"courseName\": \"Cloud Computing\"}")));
        assertEquals("1st Oct 2026", tools.execute(call("2", "get_batch_dates", "{\"course_name\": \"Data Science\"}")));
        assertTrue(tools.execute(call("3", "nope", "{}")).startsWith("Error: unknown tool"));
    }

    @Test
    void requestStepReturnsWhatTheModelAskedFor() {
        var llm = new ScriptedChatModel(AiMessage.from(List.of(
                call("c1", "get_course_fee", "{\"courseName\":\"Data Science\"}"),
                call("c2", "get_batch_dates", "{\"courseName\":\"Data Science\"}"))));

        List<ToolExecutionRequest> calls = ToolCallRequestMain.requestedToolCalls(llm, ToolCallRequestMain.QUESTION);

        assertEquals(List.of("get_course_fee", "get_batch_dates"), calls.stream().map(ToolExecutionRequest::name).toList());
        assertEquals(2, llm.requests.getFirst().toolSpecifications().size(), "both tools are sent with the question");
    }

    @Test
    void loopRunsTheToolsAndAsksAgainWithTheResults() {
        var llm = new ScriptedChatModel(
                AiMessage.from(List.of(call("c1", "get_course_fee", "{\"courseName\":\"Data Science\"}"),
                        call("c2", "get_batch_dates", "{\"courseName\":\"Data Science\"}"))),
                AiMessage.from("The fee is INR 75,000 and the next batch starts on 1st October 2026."));

        List<ChatMessage> messages = ToolCallLoopMain.run(llm, new CourseTools(), ToolCallRequestMain.QUESTION);

        // human, ai(2 tool calls), tool, tool, ai(final)
        assertEquals(5, messages.size());
        assertEquals("INR 75000", ((ToolExecutionResultMessage) messages.get(2)).text());
        assertEquals("c2", ((ToolExecutionResultMessage) messages.get(3)).id(), "result is linked to its call id");
        assertTrue(((AiMessage) messages.getLast()).text().contains("INR 75,000"));
        assertEquals(4, llm.requests.get(1).messages().size(), "the second call sees the question, calls and results");
    }
}

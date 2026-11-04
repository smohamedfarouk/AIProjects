package com.bepec.mcp;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests. The integration tests start the real MCP servers as child processes over stdio, but use a
 * scripted fake model instead of OpenAI, so they are free to run and need no API key or internet.
 */
class McpTest {

    @Test
    void mathToolsWork() {
        assertEquals(8.0, MathServerMain.add(3, 5));
        assertEquals(96.0, MathServerMain.multiply(8, 12));
        assertTrue(MathServerMain.currentTime().matches("\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}"));
    }

    @Test
    void weatherHelpersWork() {
        assertEquals("Overcast", WeatherServerMain.describe(3));
        assertEquals("Unknown", WeatherServerMain.describe(1234));
        assertEquals(1, WeatherServerMain.clampDays(0));
        assertEquals(7, WeatherServerMain.clampDays(30));
        assertEquals(3, WeatherServerMain.clampDays(3));
    }

    @Test
    void toolsAreDiscoveredAndCalledOverMcp() {
        try (var client = new MultiServerMcpClient(Map.of(
                "beginner_tools", MathServerMain.class, "weather_tools", WeatherServerMain.class))) {

            List<String> names = client.getTools().stream().map(ToolSpecification::name).sorted().toList();
            assertEquals(List.of("add", "current_time", "get_current_weather", "get_forecast", "multiply"), names);

            assertEquals("96.0", client.execute(call("1", "multiply", "{\"a\": 8, \"b\": 12}")));
            // missing "b": rejected by the tool's JSON schema; returned as text so the model can retry
            String error = client.execute(call("2", "add", "{\"a\": 3}"));
            assertTrue(error.startsWith("Error:") && error.contains("'b'"), error);
            assertTrue(client.execute(call("3", "nope", "{}")).contains("unknown tool"));
        }
    }

    @Test
    void agentLoopRunsToolsUntilFinalAnswer() {
        // Scripted model: same plan a real model usually follows for the demo question.
        List<AiMessage> script = List.of(
                AiMessage.from(List.of(call("c1", "add", "{\"a\": 3, \"b\": 5}"), call("c2", "current_time", "{}"))),
                AiMessage.from(List.of(call("c3", "multiply", "{\"a\": 8, \"b\": 12}"))),
                AiMessage.from("(3 + 5) x 12 = 96."));
        List<ChatRequest> seen = new ArrayList<>();
        ChatModel fake = new ChatModel() {
            @Override
            public ChatResponse doChat(ChatRequest request) {
                seen.add(request);
                return ChatResponse.builder().aiMessage(script.get(seen.size() - 1)).build();
            }
        };

        try (var client = new MultiServerMcpClient(Map.of("beginner_tools", MathServerMain.class))) {
            List<ChatMessage> messages = AgentMain.run(fake, client, client.getTools(), AgentMain.QUESTION);

            // human, ai(2 calls), tool, tool, ai(1 call), tool, ai(final)
            assertEquals(7, messages.size());
            assertEquals("8.0", ((ToolExecutionResultMessage) messages.get(2)).text());
            assertEquals("96.0", ((ToolExecutionResultMessage) messages.get(5)).text());
            assertEquals("(3 + 5) x 12 = 96.", ((AiMessage) messages.getLast()).text());

            // every model call carried the MCP tools, and the third call saw all earlier tool results
            assertTrue(seen.stream().allMatch(r -> r.toolSpecifications().size() == 3));
            assertEquals(6, seen.get(2).messages().size());

            assertTrue(AgentMain.prettyPrint(messages.get(1)).contains("Tool Calls:"));
        }
    }

    private static ToolExecutionRequest call(String id, String name, String json) {
        return ToolExecutionRequest.builder().id(id).name(name).arguments(json).build();
    }
}

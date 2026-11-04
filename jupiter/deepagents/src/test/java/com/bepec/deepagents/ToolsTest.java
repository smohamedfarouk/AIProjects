package com.bepec.deepagents;

import com.bepec.deepagents.agent.AgentResult;
import com.bepec.deepagents.agent.DeepAgent;
import com.bepec.deepagents.agent.SubAgent;
import com.bepec.deepagents.tools.ClassifyIntentTool;
import com.bepec.deepagents.tools.ExpandKeywordsTool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToolsTest {

    @Test
    void expandKeywords() {
        List<String> out = new ExpandKeywordsTool().expandKeywords("ai course");
        assertEquals(8, out.size());
        assertEquals("ai course course", out.get(0));
        assertEquals("ai course 2026", out.get(7));
    }

    @Test
    void classifyIntent() {
        var tool = new ClassifyIntentTool();
        assertEquals("informational", tool.classifyIntent("free AI course").get("intent"));
        assertEquals("high", tool.classifyIntent("buy AI course").get("ad_priority"));
        assertEquals("navigational", tool.classifyIntent("AI training online").get("intent"));
    }

    /** Scripted model: supervisor delegates to a subagent which writes a file, then both answer. */
    @Test
    void supervisorDelegatesToSubagentWithSharedFiles() {
        Deque<AiMessage> script = new ArrayDeque<>(List.of(
                AiMessage.from(call("1", "task",
                        "{\"description\":\"expand 'ai'\",\"subagent_type\":\"expansion-agent\"}")),
                AiMessage.from(call("2", "expand_keywords", "{\"seed\":\"ai\"}")),
                AiMessage.from(call("3", "write_file", "{\"path\":\"expanded.md\",\"content\":\"ai course\"}")),
                AiMessage.from("subagent done"),
                AiMessage.from(call("4", "read_file", "{\"path\":\"expanded.md\"}")),
                AiMessage.from("final: ai course")));

        ChatModel fake = new ChatModel() {
            @Override
            public ChatResponse doChat(ChatRequest request) {
                List<String> names = request.toolSpecifications().stream().map(ToolSpecification::name).toList();
                ChatMessage system = request.messages().get(0);
                boolean isSub = ((SystemMessage) system).text().startsWith("Read 'seeds.md'");
                assertEquals(!isSub, names.contains("task"), "only the supervisor gets the task tool");
                return ChatResponse.builder().aiMessage(script.pop()).build();
            }
        };

        DeepAgent supervisor = DeepAgent.builder()
                .model(fake)
                .systemPrompt("supervisor")
                .subagents(new SubAgent("expansion-agent", "expands", "Read 'seeds.md' ...",
                        List.of(new ExpandKeywordsTool())))
                .build();

        AgentResult out = supervisor.invoke("go");
        assertEquals("final: ai course", out.lastText());
        assertEquals("ai course", out.files().get("expanded.md"));
        assertTrue(script.isEmpty());
    }

    private static ToolExecutionRequest call(String id, String name, String args) {
        return ToolExecutionRequest.builder().id(id).name(name).arguments(args).build();
    }
}

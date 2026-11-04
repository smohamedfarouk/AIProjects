package com.bepec.chatbot;

import dev.langchain4j.data.message.ChatMessage;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * Java port of frontend.py's server side: one chat memory per browser session,
 * a chat endpoint, and a clear-conversation endpoint.
 */
@RestController
@RequestMapping("/api")
public class ChatController {

    private static final String MEMORY_KEY = "memory";

    private final ChatbotService chatbot;

    public ChatController(ChatbotService chatbot) {
        this.chatbot = chatbot;
    }

    public record ChatRequest(String message) {}

    public record ChatResponse(String reply) {}

    @PostMapping("/chat")
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request, HttpSession session) {
        if (request == null || request.message() == null || request.message().isBlank()) {
            return ResponseEntity.badRequest().build();
        }
        String reply = chatbot.converse(request.message(), memory(session));
        return ResponseEntity.ok(new ChatResponse(reply));
    }

    @PostMapping("/clear")
    public Map<String, String> clear(HttpSession session) {
        session.setAttribute(MEMORY_KEY, chatbot.newMemory());
        return Map.of("status", "cleared");
    }

    @SuppressWarnings("unchecked")
    private List<ChatMessage> memory(HttpSession session) {
        synchronized (session) {
            List<ChatMessage> memory = (List<ChatMessage>) session.getAttribute(MEMORY_KEY);
            if (memory == null) {
                memory = chatbot.newMemory();
                session.setAttribute(MEMORY_KEY, memory);
            }
            return memory;
        }
    }
}

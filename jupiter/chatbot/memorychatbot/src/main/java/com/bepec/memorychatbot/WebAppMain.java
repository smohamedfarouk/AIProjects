package com.bepec.memorychatbot;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.openai.OpenAiTokenCountEstimator;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

/**
 * BEPEC Memory Chatbot - main class. Java port of frontend.py (Streamlit) + backend.py (LangChain).
 * <p>
 * Run: {@code mvn compile exec:java} (needs OPENAI_API_KEY in the environment or a .env file),
 * then open http://localhost:8501. An optional first argument changes the port.
 *
 * <h2>Key idea: the LLM has no memory - the program does</h2>
 * OpenAI's model remembers nothing between calls. Every call is independent. "Memory" means this
 * program stores the conversation and sends the recent part back to the model with every new question.
 *
 * <h2>What happens on each message</h2>
 * <pre>
 * Browser (index.html)          WebAppMain                 ChatbotBackend                 OpenAI
 * --------------------          ------                     --------------                 ------
 * type "What is my name?"  -->  POST /api/chat
 *                               read cookie BEPEC_SESSION
 *                               -> find this browser's
 *                                  ChatHistory        -->  build the prompt:
 *                                                           1. system prompt
 *                                                           2. 3 few-shot examples
 *                                                           3. trimmed past messages
 *                                                           4. the new question    -->  gpt-4o-mini
 *                                                                                        reads it all
 *                                                          save question + reply  <--  "Your name is Asha..."
 *                                                          into ChatHistory
 * show reply          <------   {"reply": "..."}
 * </pre>
 *
 * <h2>1. One memory per browser (this class + {@link ChatHistory})</h2>
 * On the first visit the server gives the browser a random id in the cookie {@code BEPEC_SESSION}.
 * {@link #sessions} maps each id to its own {@link ChatHistory} (a list of messages), so different
 * browsers get separate conversations. This replaces Streamlit's {@code st.session_state.memory}.
 *
 * <h2>2. The prompt sent every turn ({@link ChatbotBackend#buildPrompt})</h2>
 * <pre>
 * | Part              | Purpose                                                              |
 * |-------------------|----------------------------------------------------------------------|
 * | System prompt     | Role and rules: friendly BEPEC assistant, honest, ends with          |
 * |                   | "Thanks for Asking BEPEC"                                            |
 * | Few-shot examples | 3 made-up question/answer pairs showing the tone and style to copy   |
 * | History           | Earlier messages of this session - this part IS the "memory"         |
 * | New question      | What the user just typed                                             |
 * </pre>
 *
 * <h2>3. Saving the turn ({@link ChatbotBackend#converse})</h2>
 * After the reply arrives, the question and the reply are added to the session's
 * {@link ChatHistory}. If the call fails, the user sees "Sorry, something went wrong..." and nothing
 * is saved, so the history stays clean.
 *
 * <h2>4. Keeping the prompt small ({@link MessageTrimmer})</h2>
 * Long chats would make every request bigger, slower and more expensive. Before each call the trimmer
 * starts from the newest message and works backwards, adding up tokens (OpenAI's own token counter),
 * stops at 1000 tokens, and drops leading AI messages so the kept part starts with a user message.
 * The full history stays in {@link ChatHistory}; trimming only limits what is sent, so the bot slowly
 * "forgets" the oldest parts of a long chat.
 *
 * <h2>5. Clear Conversation</h2>
 * The button calls {@code POST /api/clear}, which empties this session's {@link ChatHistory}, so the
 * bot starts fresh.
 *
 * <h2>Example</h2>
 * <ol>
 *   <li>"My name is Asha." - the history is empty; the model sees the system prompt, the examples and
 *       this message. The question and reply are saved.</li>
 *   <li>"What is my name?" - the prompt now also includes turn 1, so the model can answer "Asha".</li>
 *   <li>Click Clear, ask again - the history is gone, so the model does not know the name.</li>
 * </ol>
 *
 * <h2>Limits</h2>
 * Memory lives only in the server's RAM: restarting the server loses every conversation. Keeping chats
 * across restarts would need storage such as a file or database. Replies arrive whole, not streamed.
 *
 * <h2>HTTP endpoints served by this class</h2>
 * <pre>
 *   GET  /              chat page (src/main/resources/static/index.html)
 *   GET  /api/history   messages of this session
 *   POST /api/chat      {"message": "..."} -> {"reply": "..."}
 *   POST /api/clear     "Clear Conversation"
 * </pre>
 *
 * <h2>Classes in this project</h2>
 * <pre>
 * | Class          | What it does                                        | Python equivalent             |
 * |----------------|-----------------------------------------------------|-------------------------------|
 * | WebAppMain     | Main class: web server, sessions, chat API          | frontend.py (Streamlit)       |
 * | Config         | Loads .env, builds the OpenAI model (gpt-4o-mini,   | load_dotenv() + demo_chatbot()|
 * |                | temperature 0.3, max 1024 tokens)                   |                               |
 * | ChatbotBackend | System prompt, few-shot examples, one chat turn     | CHAT_PROMPT, demo_conversation|
 * | ChatHistory    | Full message list for one session                   | InMemoryChatMessageHistory    |
 * | MessageTrimmer | Keeps the newest 1000 tokens, starting on a user    | trim_messages(...)            |
 * |                | message                                             |                               |
 * </pre>
 * The comment covers:
 *
 *   - The key idea: the AI model remembers nothing between calls, so the program stores the conversation and sends the recent part back with each new question.
 *   - What happens on each message: a flow diagram from browser to WebAppMain to ChatbotBackend to OpenAI and back.
 *   - The five parts: one memory per browser (cookie), the 4-part prompt, saving each turn, trimming to 1000 tokens, and Clear Conversation.
 *   - The "My name is Asha" example.
 *   - Limits: memory is lost when the server restarts, and replies aren't streamed.
 *   - Reference: the web addresses the server answers and a table of every class with its Python equivalent.
 *
 */
public final class WebAppMain {

    private static final String COOKIE = "BEPEC_SESSION";
    private static final ObjectMapper JSON = new ObjectMapper();

    private final ChatbotBackend backend;
    private final Map<String, ChatHistory> sessions = new ConcurrentHashMap<>();

    WebAppMain(ChatbotBackend backend) {
        this.backend = backend;
    }

    /**
     * Entry point - starts the chatbot web server (the Java version of {@code streamlit run frontend.py}).
     * <ol>
     *   <li><b>Pick the port</b> - 8501 (Streamlit's default) unless a different one is passed as the first
     *       argument, e.g. {@code mvn exec:java -Dexec.args=9000}.</li>
     *   <li><b>Build the trimmer</b> - {@link MessageTrimmer} with a 1000-token budget, counting tokens exactly like
     *       OpenAI does for gpt-4o-mini ({@code OpenAiTokenCountEstimator}).</li>
     *   <li><b>Build the backend</b> - {@link ChatbotBackend} with the OpenAI model from {@link Config#llm()}
     *       (fails right away with a clear message if OPENAI_API_KEY is missing).</li>
     *   <li><b>Start the HTTP server</b> - the JDK's built-in {@link HttpServer}, listening on localhost only. Every
     *       request goes to {@link #handle}; each request runs on its own virtual thread, so one slow OpenAI call
     *       does not block other users.</li>
     * </ol>
     * The server keeps running until you stop the program (Ctrl+C). All conversations are held in memory and are
     * lost when it stops.
     *
     * @param args optional: the port number
     */
    public static void main(String[] args) throws IOException {
        int port = Integer.parseInt(args.length > 0 ? args[0] : "8501");
        var trimmer = new MessageTrimmer(1000, new OpenAiTokenCountEstimator(Config.MODEL_NAME));
        var app = new WebAppMain(new ChatbotBackend(Config.llm(), trimmer));

        HttpServer server = HttpServer.create(new InetSocketAddress("localhost", port), 0);
        server.createContext("/", app::handle);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        System.out.println("BEPEC Memory Chatbot running at http://localhost:" + port);
    }

    /**
     * Routes every HTTP request by method + path (a tiny hand-written router):
     * {@code GET /} -> the chat page, {@code GET /api/history} -> this session's messages (used when the page loads),
     * {@code POST /api/chat} -> {@link #chat}, {@code POST /api/clear} -> empty this session's history,
     * anything else -> 404. Unexpected errors become a JSON {@code {"error": ...}} with status 500, and
     * try-with-resources always closes the exchange so the browser never waits forever.
     */
    private void handle(HttpExchange ex) throws IOException {
        try (ex) {
            String path = ex.getRequestURI().getPath();
            String method = ex.getRequestMethod();
            switch (method + " " + path) {
                case "GET /" -> sendPage(ex);
                case "GET /api/history" -> sendJson(ex, 200, Map.of("messages", toView(session(ex).messages())));
                case "POST /api/chat" -> chat(ex);
                case "POST /api/clear" -> {
                    session(ex).clear();
                    sendJson(ex, 200, Map.of("ok", true));
                }
                default -> sendJson(ex, 404, Map.of("error", "Not found"));
            }
        } catch (Exception e) {
            sendJson(ex, 500, Map.of("error", String.valueOf(e.getMessage())));
        }
    }

    /**
     * Handles one chat message from the page.
     * <ol>
     *   <li>Read the JSON body {@code {"message": "..."}}; an empty message is rejected with 400.</li>
     *   <li>Find this browser's {@link ChatHistory} via the session cookie ({@link #session}).</li>
     *   <li>{@link ChatbotBackend#converse} builds the prompt (system + examples + trimmed history + message), calls
     *       OpenAI, and saves the question and reply into the history - this is where the "memory" happens.</li>
     *   <li>Send back {@code {"reply": "..."}}; the page shows it under the user's message.</li>
     * </ol>
     */
    private void chat(HttpExchange ex) throws IOException {
        JsonNode body = JSON.readTree(ex.getRequestBody());
        String message = body.path("message").asText("").strip();
        if (message.isEmpty()) {
            sendJson(ex, 400, Map.of("error", "message is required"));
            return;
        }
        String reply = backend.converse(message, session(ex));
        sendJson(ex, 200, Map.of("reply", reply));
    }

    /**
     * Finds this browser's conversation memory - the Java version of {@code st.session_state.memory}.
     * <ol>
     *   <li>Look for the {@code BEPEC_SESSION} cookie in the request.</li>
     *   <li>No cookie, or an id the server does not know (e.g. after a restart): create a new random id and send
     *       it back with {@code Set-Cookie}; the browser returns it on every later request.</li>
     *   <li>Return the {@link ChatHistory} stored for that id, creating an empty one the first time.</li>
     * </ol>
     * The cookie is HttpOnly (page scripts cannot read it) and SameSite=Strict (other sites cannot use it).
     */
    private ChatHistory session(HttpExchange ex) {
        String id = null;
        for (String header : ex.getRequestHeaders().getOrDefault("Cookie", List.of())) {
            for (String part : header.split(";")) {
                String[] kv = part.strip().split("=", 2);
                if (kv.length == 2 && kv[0].equals(COOKIE)) {
                    id = kv[1];
                }
            }
        }
        if (id == null || !sessions.containsKey(id)) {
            id = UUID.randomUUID().toString();
            ex.getResponseHeaders().add("Set-Cookie", COOKIE + "=" + id + "; Path=/; HttpOnly; SameSite=Strict");
        }
        return sessions.computeIfAbsent(id, k -> ChatbotBackend.newMemory());
    }

    /**
     * Converts the stored LangChain4j messages into simple {role, text} pairs for the page, so a reload of the
     * browser redraws the conversation. User messages become "user", AI messages "assistant".
     */
    private static List<Map<String, String>> toView(List<ChatMessage> messages) {
        List<Map<String, String>> view = new ArrayList<>();
        for (ChatMessage m : messages) {
            if (m instanceof UserMessage u) {
                view.add(Map.of("role", "user", "text", u.singleText()));
            } else if (m instanceof AiMessage a && a.text() != null) {
                view.add(Map.of("role", "assistant", "text", a.text()));
            }
        }
        return view;
    }

    /** Sends the chat page (src/main/resources/static/index.html, read from the classpath) and sets the cookie. */
    private void sendPage(HttpExchange ex) throws IOException {
        session(ex); // set the cookie early
        try (InputStream in = WebAppMain.class.getResourceAsStream("/static/index.html")) {
            if (in == null) {
                throw new IOException("index.html missing from classpath");
            }
            send(ex, 200, "text/html; charset=utf-8", in.readAllBytes());
        }
    }

    /** Writes any Java object (usually a Map) as a JSON response with the given HTTP status. */
    private static void sendJson(HttpExchange ex, int status, Object body) throws IOException {
        send(ex, status, "application/json; charset=utf-8", JSON.writeValueAsBytes(body));
    }

    private static void send(HttpExchange ex, int status, String type, byte[] bytes) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(status, bytes.length);
        ex.getResponseBody().write(bytes);
    }
}

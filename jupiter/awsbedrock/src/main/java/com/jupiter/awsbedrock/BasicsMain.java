package com.jupiter.awsbedrock;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import software.amazon.awssdk.core.document.Document;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.*;

/**
 * Five progressive AWS Bedrock Converse examples using one pricing scenario.
 * Java 21 / AWS SDK v2. Tools run locally in Java, not inside Bedrock.
 *
 * <h2>Run</h2>
 * {@code mvn compile exec:java}. Needs AWS credentials from the default credential chain (environment variables,
 * {@code ~/.aws/credentials}, an AWS SSO / CLI profile, ...) and access to the model amazon.nova-lite-v1:0 enabled in the
 * Bedrock console for region us-east-1. Every cell makes real, billed Bedrock calls.
 *
 * <h2>The scenario</h2>
 * "An order costs GBP 120. A student gets a 10 percent discount. How much is saved, and what is the final amount?"
 * The correct answer is: saved GBP 12.00, final amount GBP 108.00. Each cell answers the same question in a more
 * reliable way, so you can compare the approaches.
 *
 * <h2>Key ideas</h2>
 * <ul>
 *   <li><b>Converse API</b> - one request format for every Bedrock chat model: a list of messages (role + content
 *       blocks), an optional system prompt, inference settings, and optional tools. Switching models only means
 *       changing the model id.</li>
 *   <li><b>Content blocks</b> - a message holds blocks: text, a tool-use request (from the model) or a tool result
 *       (from us). That is why the code loops over {@code content()} instead of reading a single string.</li>
 *   <li><b>Tool calling</b> - the model cannot run code. It replies with a {@code toolUse} block (tool name + JSON
 *       input) and stop reason {@code TOOL_USE}; <i>Java</i> runs the tool and sends the answer back as a
 *       {@code toolResult} block; the model then continues. Money maths is done with {@link BigDecimal}, so the
 *       numbers are exact instead of the model's "mental arithmetic".</li>
 *   <li><b>The model has no memory</b> - every call sends the whole conversation ({@code history}) again.</li>
 * </ul>
 *
 * <h2>The tool-calling loop (cells 4 and 5)</h2>
 * <pre>
 *   history = [user question]
 *   repeat (max turns):
 *       response = converse(history, tools)           // invokeModel
 *       history += model reply
 *       stop reason != TOOL_USE ?  print answer, stop   // runToolLoop
 *       for each toolUse block: run Java tool           // processToolRequests / executeTool
 *       history += user message with toolResult blocks  // matched by toolUseId
 * </pre>
 */
public class BasicsMain {
    private static final String MODEL = "amazon.nova-lite-v1:0";
    private static final String ORDER_QUESTION =
            "An order costs GBP 120. A student gets a 10 percent discount. "
                    + "How much is saved, and what is the final amount?";
    /**
     * Runs all five cells in order against Amazon Nova Lite, printing a header before each one.
     * <ol>
     *   <li><b>Create the client</b> - {@link BedrockRuntimeClient} for region us-east-1. It finds AWS credentials
     *       automatically (default credential chain). try-with-resources closes its HTTP connections at the end.</li>
     *   <li><b>Run the cells</b>, each building on the previous one:</li>
     * </ol>
     * <pre>
    Cell 1 — Simple LLM
    Sends the £120 discount question directly to Amazon Nova Lite. The model generates the answer without Java calculation tools.

    Cell 2 — System prompt + user prompt
    Same question, but adds a system instruction to behave as a retail pricing assistant and explain the calculation clearly.

    Cell 3 — Raw Converse API response
    Same question again, but demonstrates how to extract the answer, token usage and stop reason from ConverseResponse.

    Cell 4 — Java tool calling
    The model requests apply_discount, Java uses BigDecimal to calculate the saving and final price, and the tool result is sent back to Bedrock.

    Cell 5 — Multi-step AI agent
    The model first requests get_demo_discount to obtain the student discount, then requests apply_discount to calculate the final price. Finally, it explains the result.
     * </pre>
     * If a cell fails (no credentials, model access not enabled, wrong region) the exception stops the program and
     * the remaining cells do not run.
     *
     * @param args not used
     */
    public static void main(String[] args) {
        try (BedrockRuntimeClient client = BedrockRuntimeClient.builder()
                .region(Region.US_EAST_1)
                .build()) {

            System.out.println("\n===== CELL 1: SIMPLE LLM PROMPT =====");
            cell1SimplePrompt(client);

            System.out.println("\n===== CELL 2: SYSTEM + USER PROMPT =====");
            cell2SystemPrompt(client);

            System.out.println("\n===== CELL 3: RAW CONVERSE RESPONSE =====");
            cell3RawResponse(client);

            System.out.println("\n===== CELL 4: JAVA CALCULATION TOOL =====");
            cell4ToolCalling(client);

            System.out.println("\n===== CELL 5: MULTI-STEP AGENT =====");
            cell5MultiStepAgent(client);
        }
    }

    /**
     * Cell 1 - the simplest possible call: one user message, no system prompt, no tools.
     * <p>
     * Sends {@link #ORDER_QUESTION} with temperature 0.2 (low randomness, so answers are fairly stable) and at most
     * 400 output tokens, then prints the text of the reply. The model works out the discount on its own - usually
     * right for easy numbers like these, but nothing guarantees the arithmetic. Cells 4 and 5 fix that.
     */
    static void cell1SimplePrompt(BedrockRuntimeClient client) {
        ConverseResponse response = client.converse(r -> r
                .modelId(MODEL)
                .messages(userMessage(ORDER_QUESTION))
                .inferenceConfig(config(0.2f, 400)));

        System.out.println("AI: " + textOf(response));
    }

    /**
     * Cell 2 - the same question plus a <b>system prompt</b>.
     * <p>
     * The system prompt is an instruction that sits above the conversation and shapes every answer: here it makes
     * the model act as a retail pricing assistant and always show subtotal, discount percentage, amount saved and
     * final price, in GBP, briefly. Compare the output with cell 1 to see how much the format changes.
     */
    static void cell2SystemPrompt(BedrockRuntimeClient client) {
        ConverseResponse response = client.converse(r -> r
                .modelId(MODEL)
                .system(SystemContentBlock.fromText(
                        "You are a retail pricing assistant. Show the subtotal, "
                                + "discount percentage, amount saved, and final price. "
                                + "Use GBP and keep the explanation short."))
                .messages(userMessage(ORDER_QUESTION))
                .inferenceConfig(config(0.2f, 400)));

        System.out.println("AI: " + textOf(response));
    }

    /**
     * Cell 3 - the same request as cell 1, but showing what is inside a {@link ConverseResponse}.
     * <ul>
     *   <li>{@code output().message()} - the assistant's message; its {@code content()} is a list of blocks, and
     *       only text blocks are printed.</li>
     *   <li>{@code stopReason()} - why the model stopped: {@code END_TURN} (finished normally),
     *       {@code MAX_TOKENS} (cut off by the 400-token limit) or {@code TOOL_USE} (wants a tool - see cell 4).</li>
     *   <li>{@code usage()} - input and output token counts; Bedrock bills by these.</li>
     * </ul>
     */
    static void cell3RawResponse(BedrockRuntimeClient client) {
        ConverseResponse response = client.converse(r -> r
                .modelId(MODEL)
                .messages(userMessage(ORDER_QUESTION))
                .inferenceConfig(config(0.2f, 400)));

        Message assistant = response.output().message();
        for (ContentBlock block : assistant.content()) {
            if (block.text() != null) {
                System.out.println("AI: " + block.text());
            }
        }
        System.out.println("Stop reason: " + response.stopReason());
        System.out.println("Input tokens: " + response.usage().inputTokens());
        System.out.println("Output tokens: " + response.usage().outputTokens());
    }

    /**
     * Cell 4 - <b>tool calling</b>: instead of trusting the model's mental arithmetic, force one Java tool call.
     * <ol>
     *   <li>The model is offered one tool, apply_discount, and on turn 1 is required to call it.</li>
     *   <li>It replies with a toolUse block, e.g. {@code {"subtotal": 120, "discount_percent": 10}}.</li>
     *   <li>Java calculates exactly with BigDecimal: amount_saved 12.00, final_price 108.00.</li>
     *   <li>The result goes back to the model, which writes the final answer from those exact numbers.</li>
     * </ol>
     * Allows up to 4 turns; normally it takes 2 (tool call, then answer).
     */
    static void cell4ToolCalling(BedrockRuntimeClient client) {
        runToolLoop(
                client,
                ORDER_QUESTION + " Use apply_discount to calculate accurately.",
                List.of(applyDiscountTool()),
                List.of("apply_discount"),
                4);
    }

    /**
     * Cell 5 - a small <b>multi-step agent</b>: the question no longer says "10 percent", only "student membership".
     * <ol>
     *   <li>Turn 1 (forced): get_demo_discount({"membership": "student"}) -> {"discount_percent": 10}.</li>
     *   <li>Turn 2 (forced): apply_discount with subtotal 120 and the 10 it just received -> 12.00 saved,
     *       108.00 final.</li>
     *   <li>Turn 3 (free): the model explains the result in words.</li>
     * </ol>
     * This shows the core agent pattern: the output of one tool becomes the input of the next, and the model decides
     * how to connect them. Allows up to 6 turns.
     */
    static void cell5MultiStepAgent(BedrockRuntimeClient client) {
        runToolLoop(
                client,
                "An order costs GBP 120. The customer has student membership. "
                        + "First look up the membership discount using get_demo_discount. "
                        + "Then calculate the saving and final price using apply_discount. "
                        + "Explain your final answer.",
                List.of(getDiscountTool(), applyDiscountTool()),
                List.of("get_demo_discount", "apply_discount"),
                6);
    }

    /**
     * Defines the apply_discount tool that the model may call: two required numbers, subtotal and
     * discount_percent. The model only sees this description - the actual calculation lives in {@link #executeTool}.
     */
    private static Tool applyDiscountTool() {
        return createTool(
                "apply_discount",
                "Calculate the amount saved and final price for a subtotal "
                        + "and discount percentage.",
                Map.of("subtotal", "number", "discount_percent", "number"));
    }

    /**
     * Defines the get_demo_discount tool: takes a membership name (string) and returns its discount percentage.
     * The description tells the model the demo values, and that this is example data, not a live store.
     */
    private static Tool getDiscountTool() {
        return createTool(
                "get_demo_discount",
                "Look up a demo membership discount: student = 10 percent, "
                        + "standard = 0 percent. This is example data, not a live store.",
                Map.of("membership", "string"));
    }

    /**
     * Builds a Bedrock {@link Tool} definition from a simple field map.
     * <p>
     * Bedrock describes tool inputs with JSON Schema. For example, apply_discount becomes:
     * <pre>
     * { "type": "object",
     *   "properties": { "subtotal": {"type": "number"}, "discount_percent": {"type": "number"} },
     *   "required": ["subtotal", "discount_percent"] }
     * </pre>
     * The AWS SDK represents JSON with {@link Document}, so the schema is built from Document maps and lists.
     * Every field is marked required.
     *
     * @param name        tool name the model uses to call it
     * @param description what the tool does; the model reads this to decide when to use it
     * @param fields      field name -> JSON type ("number", "string", ...)
     * @return the tool definition to put in the request's tool configuration
     */
    private static Tool createTool(String name, String description,
                                   Map<String, String> fields) {
        Map<String, Document> properties = new LinkedHashMap<>();
        List<Document> required = new ArrayList<>();

        for (Map.Entry<String, String> field : fields.entrySet()) {
            properties.put(field.getKey(), Document.fromMap(Map.of(
                    "type", Document.fromString(field.getValue()))));
            required.add(Document.fromString(field.getKey()));
        }

        Document schema = Document.fromMap(Map.of(
                "type", Document.fromString("object"),
                "properties", Document.fromMap(properties),
                "required", Document.fromList(required)));

        return Tool.builder()
                .toolSpec(ToolSpecification.builder()
                        .name(name)
                        .description(description)
                        .inputSchema(ToolInputSchema.fromJson(schema))
                        .build())
                .build();
    }

    /**
     * The orchestration loop: ask -> tool use -> execute -> reply, repeated until the model gives a final answer.
     * <ol>
     *   <li>Start the history with the user's question.</li>
     *   <li>Each turn: call the model with the full history ({@link #invokeModel}) and append its reply.</li>
     *   <li>If the stop reason is not TOOL_USE, the model has answered: print it and stop.</li>
     *   <li>Otherwise run every requested tool in Java ({@link #processToolRequests}) and append the results as a new
     *       user message, then loop so the model can read them.</li>
     * </ol>
     * The first prescribed tools ({@code plannedTools}) make this teaching example repeatable: turn N is forced to
     * call tool N. After those steps the model is free to produce its final answer. {@code maxTurns} is a safety
     * limit so a model that keeps calling tools cannot loop forever.
     *
     * @param client       Bedrock client
     * @param question     the user's question
     * @param tools        tools the model may call
     * @param plannedTools tool forced on each early turn, in order (turn 1 -> first entry, ...)
     * @param maxTurns     maximum number of model calls
     */
    private static void runToolLoop(
            BedrockRuntimeClient client,
            String question,
            List<Tool> tools,
            List<String> plannedTools,
            int maxTurns) {

        List<Message> history = new ArrayList<>();
        history.add(userMessage(question));

        for (int turn = 1; turn <= maxTurns; turn++) {

            System.out.println("Agent turn: " + turn);

            ConverseResponse response = invokeModel(
                    client, history, tools, plannedTools, turn);

            Message assistantMessage = response.output().message();
            history.add(assistantMessage);

            if (response.stopReason() != StopReason.TOOL_USE) {
                printFinalResponse(response);
                return;
            }

            Message toolResults = processToolRequests(assistantMessage);
            history.add(toolResults);
        }

        System.out.println("Stopped at the maximum agent turn count.");
    }

    /**
     * One model call inside the tool loop.
     * <p>
     * Sends the whole history (the model remembers nothing between calls), a system prompt that says "use tool
     * results, never invent them", the tool configuration for this turn ({@link #buildToolConfiguration}) and
     * temperature 0.1 for very consistent behaviour, with up to 700 output tokens.
     *
     * @return the model's response: either text (finished) or one or more toolUse blocks
     */
    private static ConverseResponse invokeModel(
            BedrockRuntimeClient client,
            List<Message> history,
            List<Tool> tools,
            List<String> plannedTools,
            int turn) {

        ToolConfiguration toolConfig =
                buildToolConfiguration(tools, plannedTools, turn);

        return client.converse(r -> r
                .modelId(MODEL)
                .system(SystemContentBlock.fromText(
                        "You are an accurate shop pricing assistant. "
                                + "Use tool results for prices and discounts. "
                                + "Never make up a tool result. "
                                + "After all calculations, give a short answer."))
                .messages(history)
                .toolConfig(toolConfig)
                .inferenceConfig(config(0.1f, 700)));
    }
    /**
     * Runs every tool the model asked for in one reply and packs all results into a single message.
     * <p>
     * A reply can contain several toolUse blocks (the model may ask for more than one tool at once), mixed with text
     * blocks. Each toolUse block is executed by {@link #processSingleTool}. Bedrock expects all results for one
     * reply in the next message, which is built by {@link #createToolResultsMessage}. If the stop reason said
     * TOOL_USE but there is no toolUse block, the response is invalid, so this throws.
     *
     * @param assistantMessage the model's reply that requested tools
     * @return a user-role message holding one toolResult block per request
     */
    private static Message processToolRequests(Message assistantMessage) {

        List<ContentBlock> results = new ArrayList<>();

        for (ContentBlock block : assistantMessage.content()) {

            ToolUseBlock toolUse = block.toolUse();

            if (toolUse != null) {
                results.add(processSingleTool(toolUse));
            }
        }

        if (results.isEmpty()) {
            throw new IllegalStateException(
                    "Bedrock requested tools but provided no tool-use blocks");
        }

        return createToolResultsMessage(results);
    }

    /**
     * Runs a tool but never lets it crash the loop: any exception (bad input, unknown tool, unsupported membership)
     * is turned into {@code {"error": "..."}}. The model receives that error as the tool result and can explain it or
     * try again with different input.
     */
    private static Document executeToolSafely(ToolUseBlock toolUse) {

        try {
            return executeTool(toolUse);

        } catch (RuntimeException ex) {

            String errorMessage = ex.getMessage() == null
                    ? "Tool failed"
                    : ex.getMessage();

            return Document.fromMap(Map.of(
                    "error", Document.fromString(errorMessage)
            ));
        }
    }

    /**
     * Wraps tool results in a message with role USER. In the Converse API, tool results are sent back on the user
     * side of the conversation - they are information the model did not have, just like a user's reply.
     */
    private static Message createToolResultsMessage(
            List<ContentBlock> results) {

        return Message.builder()
                .role(ConversationRole.USER)
                .content(results)
                .build();
    }



    /**
     * Builds the tool configuration for one turn.
     * <ul>
     *   <li>Always lists every available tool, so the model knows they exist.</li>
     *   <li>For the first {@code plannedTools.size()} turns, sets {@code toolChoice} to that specific tool, which
     *       <b>forces</b> the model to call it (turn 1 -> plannedTools[0], turn 2 -> plannedTools[1], ...). This keeps
     *       the demo predictable.</li>
     *   <li>After that no tool choice is set (the default, "auto"): the model decides whether to call a tool or
     *       answer, and normally writes the final answer.</li>
     * </ul>
     */
    private static ToolConfiguration buildToolConfiguration(
            List<Tool> tools,
            List<String> plannedTools,
            int turn) {

        ToolConfiguration.Builder builder =
                ToolConfiguration.builder().tools(tools);

        if (turn <= plannedTools.size()) {
            String chosenTool = plannedTools.get(turn - 1);

            builder.toolChoice(
                    ToolChoice.fromTool(t -> t.name(chosenTool)));
        }

        return builder.build();
    }
    /**
     * Executes one toolUse request and converts the answer into a toolResult block.
     * <p>
     * Prints the tool name, its input and the Java result so you can follow the agent step by step. The result block
     * carries the same {@code toolUseId} as the request - that id is how Bedrock matches each result to the call
     * that asked for it.
     */
    private static ContentBlock processSingleTool(ToolUseBlock toolUse) {

        System.out.println("Tool requested: " + toolUse.name());
        System.out.println("Tool input: " + toolUse.input());

        Document result = executeToolSafely(toolUse);

        System.out.println("Java tool result: " + result);

        ToolResultBlock toolResult = ToolResultBlock.builder()
                .toolUseId(toolUse.toolUseId())
                .content(ToolResultContentBlock.fromJson(result))
                .build();

        return ContentBlock.fromToolResult(toolResult);
    }

    /**
     * The actual tool implementations - local Java functions: Bedrock requests them, but Java executes them.
     * <ul>
     *   <li><b>get_demo_discount</b> - reads "membership" (case-insensitive): student -> 10, standard -> 0, anything
     *       else -> error. Returns {@code {"discount_percent": n}}.</li>
     *   <li><b>apply_discount</b> - reads "subtotal" and "discount_percent", rejects negative values or a discount
     *       above 100, then calculates with BigDecimal (exact decimal maths, no floating-point errors):
     *       saving = subtotal x percent / 100, rounded half-up to 2 decimals; final = subtotal - saving.
     *       For 120 and 10: {@code {"amount_saved": 12.00, "final_price": 108.00}}.</li>
     *   <li>Any other name -> error "Tool is not allowed" (only known tools can run).</li>
     * </ul>
     * Errors are thrown here and turned into {@code {"error": ...}} results by {@link #executeToolSafely}.
     */
    private static Document executeTool(ToolUseBlock use) {
        Map<String, Document> input = use.input().asMap();

        return switch (use.name()) {
            case "get_demo_discount" -> {
                Document raw = input.get("membership");
                if (raw == null || !raw.isString()) {
                    throw new IllegalArgumentException("membership is required");
                }
                String membership = raw.asString().toLowerCase(Locale.ROOT);
                int discount = switch (membership) {
                    case "student" -> 10;
                    case "standard" -> 0;
                    default -> throw new IllegalArgumentException(
                            "Unsupported membership: " + membership);
                };
                yield Document.fromMap(Map.of(
                        "discount_percent", Document.fromNumber(discount)));
            }

            case "apply_discount" -> {
                BigDecimal subtotal = number(input, "subtotal");
                BigDecimal percent = number(input, "discount_percent");
                if (subtotal.signum() < 0 || percent.signum() < 0
                        || percent.compareTo(BigDecimal.valueOf(100)) > 0) {
                    throw new IllegalArgumentException("Invalid price or discount");
                }

                BigDecimal saving = subtotal.multiply(percent)
                        .divide(BigDecimal.valueOf(100), 2, RoundingMode.HALF_UP);
                BigDecimal finalPrice = subtotal.subtract(saving)
                        .setScale(2, RoundingMode.HALF_UP);

                yield Document.fromMap(Map.of(
                        "amount_saved", Document.fromNumber(saving),
                        "final_price", Document.fromNumber(finalPrice)));
            }

            default -> throw new IllegalArgumentException(
                    "Tool is not allowed: " + use.name());
        };
    }

    /** Reads a required numeric field from the tool input as a BigDecimal; throws if it is missing or not a number. */
    private static BigDecimal number(Map<String, Document> input, String key) {
        Document value = input.get(key);
        if (value == null || !value.isNumber()) {
            throw new IllegalArgumentException("Missing numeric field: " + key);
        }
        return value.asNumber().bigDecimalValue();
    }

    /** Creates a user-role message with one text block - the format the Converse API expects for a question. */
    private static Message userMessage(String content) {
        return Message.builder()
                .role(ConversationRole.USER)
                .content(ContentBlock.fromText(content))
                .build();
    }

    /**
     * Inference settings for a request: {@code temperature} controls randomness (0 = most predictable, higher = more
     * varied wording) and {@code maxTokens} caps the length of the reply (cost and time limit).
     */
    private static InferenceConfiguration config(float temperature, int maxTokens) {
        return InferenceConfiguration.builder()
                .temperature(temperature)
                .maxTokens(maxTokens)
                .build();
    }

    /** Joins all text blocks of the model's reply into one string, skipping non-text blocks such as toolUse. */
    private static String textOf(ConverseResponse response) {
        StringBuilder out = new StringBuilder();
        for (ContentBlock block : response.output().message().content()) {
            if (block.text() != null) {
                out.append(block.text());
            }
        }
        return out.toString();
    }
    /** Prints the model's final answer and the stop reason (normally END_TURN) when the tool loop finishes. */
    private static void printFinalResponse(ConverseResponse response) {

        System.out.println("AI: " + textOf(response));
        System.out.println("Stop reason: " + response.stopReason());
    }

}

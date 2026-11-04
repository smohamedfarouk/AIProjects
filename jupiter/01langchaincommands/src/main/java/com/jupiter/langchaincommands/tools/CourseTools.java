package com.jupiter.langchaincommands.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The two BEPEC course tools - the Java version of the notebook's {@code @tool} functions
 * {@code get_course_fee} and {@code get_batch_dates}.
 * <p>
 * {@code @Tool} plays the role of Python's {@code @tool}: LangChain4j reads the tool name, the description and the
 * parameter (name + type) and turns them into a {@link ToolSpecification} that is sent to the model
 * ({@code llm.bind_tools([...])}). The model never runs these methods itself - it only asks for them; our code runs
 * them ({@link #execute}).
 */
public final class CourseTools {

    static final Map<String, String> FEES = Map.of(
            "data science", "INR 75000",
            "ai course", "INR 80000",
            "cloud computing", "INR 85000");

    static final Map<String, String> BATCH_DATES = Map.of(
            "data science", "1st Oct 2026",
            "ai course", "1st March 2027",
            "cloud computing", "1st Oct 2026");

    private static final ObjectMapper JSON = new ObjectMapper();

    /** get the fee for a BEPEC Course. Case-insensitive: "Data Science" and "data science" both match. */
    @Tool(name = "get_course_fee", value = "get the fee for a BEPEC Course")
    public String getCourseFee(@P("course name, e.g. Data Science") String courseName) {
        return FEES.getOrDefault(courseName.toLowerCase(Locale.ROOT).strip(), "Course not found");
    }

    /** get the batch dates for a BEPEC Course. Case-insensitive. */
    @Tool(name = "get_batch_dates", value = "get the batch dates for a BEPEC Course")
    public String getBatchDates(@P("course name, e.g. Data Science") String courseName) {
        return BATCH_DATES.getOrDefault(courseName.toLowerCase(Locale.ROOT).strip(), "Course not found");
    }

    /** {@code llm.bind_tools([get_course_fee, get_batch_dates])}: the tool descriptions sent with every request. */
    public static List<ToolSpecification> specifications() {
        return ToolSpecifications.toolSpecificationsFrom(CourseTools.class);
    }

    /**
     * Runs one tool call the model asked for - the notebook's
     * {@code tools[tool_call["name"]].invoke(tool_call["args"])}.
     * <p>
     * The request carries the tool name and its arguments as JSON, e.g. {@code {"courseName": "Data Science"}}; we
     * read the argument, call the matching method and return its text, which goes back to the model as a
     * ToolMessage. An unknown tool returns an error text instead of failing, so the model can see what went wrong.
     *
     * @param request one entry of {@code response.tool_calls}
     * @return the tool's result text
     */
    public String execute(ToolExecutionRequest request) {
        String course = courseArgument(request.arguments());
        return switch (request.name()) {
            case "get_course_fee" -> getCourseFee(course);
            case "get_batch_dates" -> getBatchDates(course);
            default -> "Error: unknown tool " + request.name();
        };
    }

    private static String courseArgument(String json) {
        try {
            JsonNode args = JSON.readTree(json);
            JsonNode value = args.has("courseName") ? args.get("courseName") : args.path("course_name");
            return value.asText("");
        } catch (Exception e) {
            return "";
        }
    }
}

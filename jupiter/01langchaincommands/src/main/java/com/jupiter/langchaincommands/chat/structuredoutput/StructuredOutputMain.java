package com.jupiter.langchaincommands.chat.structuredoutput;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

/**
 * More LangChain commands, lesson 3 - structured output: get a typed Java object instead of free text
 * (morelangchaincommands.ipynb cells 8-17). <b>Entry point.</b>
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@structured-output
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>Why structured output</h2>
 * Free text is fine for people but hard for programs: you cannot reliably read "the duration" out of a paragraph.
 * With structured output the model must answer as JSON matching a schema, which is turned straight into a
 * {@link CourseInfo} - so {@code result.durationWeeks()} is an {@code int} you can use in code.
 * <pre>
 *   prompt "Generate Course Plan for a {title} ..."  --(schema of CourseInfo)-->  gpt-4o-mini
 *     --> {"title": "...", "durationWeeks": 2, "topics": [...]}  --> CourseInfo record
 * </pre>
 *
 * <h2>Python -> Java</h2>
 * <pre>
 * | Python                                         | Java                                                    |
 * |------------------------------------------------|---------------------------------------------------------|
 * | class CourseInfo(BaseModel) + Field(...)       | record CourseInfo + @Description                        |
 * | llm.with_structured_output(CourseInfo)         | Models.openAiJsonSchema(...) - strict JSON-schema mode  |
 * | prompt = ChatPromptTemplate.from_template(...) | @UserMessage on {@link CoursePlanner#plan}              |
 * | chain = prompt | struct_llm                    | AiServices.create(CoursePlanner.class, llm)             |
 * | chain.invoke({"title": ...})                   | planner.plan(title)                                     |
 * </pre>
 * Note: LangChain4j's {@code @UserMessage} templates write variables as {@code {{title}}} (double braces).
 *
 * <h2>Notebook result</h2>
 * <pre>
 * title='AWS SageMaker & Bedrock Corporate Training', duration_weeks=2,
 * topics=['Introduction to AWS SageMaker', 'Setting up AWS SageMaker Environment', ..., 'Best Practices and Case Studies']
 * </pre>
 */
public final class StructuredOutputMain {

    static final String TITLE = "AWS Sagemaker & Bedrock";

    /**
     * The chain {@code prompt | struct_llm}: LangChain4j implements this interface at runtime. Because the method
     * returns {@link CourseInfo}, it asks the model for JSON in CourseInfo's shape and converts the reply.
     */
    public interface CoursePlanner {
        @UserMessage("Generate Course Plan for a {{title}} Corporate Training of 30 Hours of duration and 4 hours "
                + "per day maximum with Implementation Approach & Custom Projects")
        CourseInfo plan(@V("title") String title);
    }

    private StructuredOutputMain() {
    }

    /** Builds the chain on any chat model (OpenAI in production, a fake one in tests). */
    static CoursePlanner planner(ChatModel llm) {
        return AiServices.create(CoursePlanner.class, llm);
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Create gpt-4o-mini in strict JSON-schema mode ({@link Models#openAiJsonSchema}).</li>
     *   <li>{@code chain.invoke({"title": "AWS Sagemaker & Bedrock"})} - returns a {@link CourseInfo}.</li>
     *   <li>Print each field separately, like the notebook's {@code result.duration_weeks}, {@code result.title} and
     *       {@code result.topics}, then the whole object.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("chain = prompt | llm.with_structured_output(CourseInfo)");
        CourseInfo result = planner(Models.openAiJsonSchema("gpt-4o-mini")).plan(TITLE);

        System.out.println("result.durationWeeks = " + result.durationWeeks());
        System.out.println("result.title         = " + result.title());
        System.out.println("result.topics:");
        result.topics().forEach(t -> System.out.println("  - " + t));
        System.out.println("\n" + result);
    }
}

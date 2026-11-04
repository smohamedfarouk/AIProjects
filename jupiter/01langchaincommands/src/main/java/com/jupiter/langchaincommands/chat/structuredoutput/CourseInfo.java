package com.jupiter.langchaincommands.chat.structuredoutput;

import dev.langchain4j.model.output.structured.Description;

import java.util.List;

/**
 * The shape the model's answer must have - the Java version of the notebook's Pydantic model:
 * <pre>
 * class CourseInfo(BaseModel):
 *     title: str = Field(description="Name of the Course")
 *     duration_weeks: int = Field(description="Duration of the course in weeks")
 *     topics: list[str] = Field(description="List of main topics covered in the course")
 * </pre>
 * LangChain4j turns this record into a JSON schema (field names, types and these descriptions) and sends it to
 * OpenAI, which then must reply with JSON that matches it. The descriptions tell the model what to put in each field.
 *
 * @param title         name of the course
 * @param durationWeeks duration of the course in weeks
 * @param topics        main topics covered in the course
 */
public record CourseInfo(
        @Description("Name of the Course") String title,
        @Description("Duration of the course in weeks") int durationWeeks,
        @Description("List of main topics covered in the course") List<String> topics) {
}

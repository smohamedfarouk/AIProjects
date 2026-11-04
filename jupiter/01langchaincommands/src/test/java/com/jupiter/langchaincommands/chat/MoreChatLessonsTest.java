package com.jupiter.langchaincommands.chat;

import com.jupiter.langchaincommands.chat.chatprompttemplate.MessagesTemplate;
import com.jupiter.langchaincommands.chat.structuredoutput.CourseInfo;
import com.jupiter.langchaincommands.chat.structuredoutput.StructuredOutputMain;
import com.jupiter.langchaincommands.testutil.Fakes.ScriptedChatModel;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.service.AiServices;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests of the chat lessons from morelangchaincommands.ipynb (no OpenAI calls). */
class MoreChatLessonsTest {

    @Test
    void messagesTemplateFillsSystemAndHumanTemplates() {
        List<ChatMessage> messages = MessagesTemplate
                .fromMessages("You are an expert in {domain}", "Explain {topic} to a {audience} in 3 lines")
                .formatMessages(Map.of("domain", "Banking", "topic", "AWS BedRock", "audience", "Senior Architects"));

        assertEquals("You are an expert in Banking", ((SystemMessage) messages.get(0)).text());
        assertEquals("Explain AWS BedRock to a Senior Architects in 3 lines", ((UserMessage) messages.get(1)).singleText());
        assertThrows(IllegalArgumentException.class, () ->
                MessagesTemplate.fromMessages("{domain}", "{topic}").formatMessages(Map.of("domain", "x")));
    }

    @Test
    void structuredOutputTurnsTheJsonReplyIntoACourseInfo() {
        var llm = new ScriptedChatModel(AiMessage.from("""
                {"title": "AWS SageMaker & Bedrock Corporate Training", "durationWeeks": 2,
                 "topics": ["Introduction to AWS SageMaker", "Introduction to AWS Bedrock"]}"""));
        StructuredOutputMain.CoursePlanner planner = AiServices.create(StructuredOutputMain.CoursePlanner.class, llm);

        CourseInfo result = planner.plan("AWS Sagemaker & Bedrock");

        assertEquals(2, result.durationWeeks());
        assertEquals("AWS SageMaker & Bedrock Corporate Training", result.title());
        assertEquals(2, result.topics().size());
        String prompt = ((UserMessage) llm.requests.getFirst().messages().getLast()).singleText();
        assertTrue(prompt.startsWith("Generate Course Plan for a AWS Sagemaker & Bedrock Corporate Training"), prompt);
    }
}

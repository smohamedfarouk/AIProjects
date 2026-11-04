package com.bepec.deepagents.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Python {@code classify_intent(keyword)}. */
public class ClassifyIntentTool {

    private static final List<String> INFORMATIONAL = List.of("free", "salary", "what is", "vs", "meaning");
    private static final List<String> COMMERCIAL = List.of("buy", "purchase", "for sale", "discount", "deal");

    @Tool(name = "classify_intent", value = """
            Classify the intent of a keywords search as commercial, informational, or navigational, and \
            assign an ad priority. use to triage keyword list""")
    public Map<String, String> classifyIntent(@P("the keyword to classify") String keyword) {
        String kw = keyword.toLowerCase();
        String intent;
        String priority;
        if (INFORMATIONAL.stream().anyMatch(kw::contains)) {
            intent = "informational";
            priority = "low";
        } else if (COMMERCIAL.stream().anyMatch(kw::contains)) {
            intent = "commercial";
            priority = "high";
        } else {
            intent = "navigational";
            priority = "medium";
        }
        Map<String, String> result = new LinkedHashMap<>();
        result.put("keyword", keyword);
        result.put("intent", intent);
        result.put("ad_priority", priority);
        return result;
    }
}

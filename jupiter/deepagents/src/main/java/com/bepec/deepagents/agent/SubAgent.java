package com.bepec.deepagents.agent;

import java.util.List;

/**
 * Java equivalent of the subagent dicts in the notebook:
 * <pre>{"name": ..., "description": ..., "system_prompt": ..., "tools": [...]}</pre>
 *
 * @param tools objects whose {@code @Tool}-annotated methods the subagent may call
 */
public record SubAgent(String name, String description, String systemPrompt, List<Object> tools) {
}

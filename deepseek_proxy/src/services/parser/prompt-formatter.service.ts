import {
  OpenAIMessage,
  OpenAITool,
  OpenAIContentPart,
} from "../../types/openai.types";

export class PromptFormatterService {
  private static readonly RULES_AND_PROTOCOL = `[RULES]
1. Always use the exact tool name and valid JSON parameters adhering to the parameters schema as in [MANDATORY TOOL PROTOCOL] block.
2. Do not simulate the tool call execution result.
3. You must use the available tools whenever they can help you to accomplish the goal.
4. NEVER claim you have modified a file, executed a command, or completed an action until you have emitted the tool call AND received the "Harness tool execution report for "tool_name"\`\`\`\n{content}\n\`\`\`" in the next turn confirming success.
5. Base your reasoning and subsequent actions STRICTLY on the real text returned in "Harness tool execution report for "tool_name"\`\`\`\n{content}\n\`\`\`", never on assumptions or imagined outcomes.
6. ALWAYS keep this in mind when providing long command sequences in a single message, because they’re not SQL transactions that can be easily rolled back on failure.
7. ALWAYS use tool calls with an absolute file/directory path or they may fail if you point relative pass e.g. "dosc/filename.extension", "dosc". Corect way to point path in tool calls is "/home/username/path/to/dir" or "/home/username/path/to/filename.extension".
8. Each tool call are responded by the harness in a proper format, knowing that you must never write response for your tool calls to not mislead the user and you. If there is no response to your tool call starting with "Harness tool execution report for "tool_name"\`\`\`\n{content}\n\`\`\`" then the tool call was not valid to be recognized by harness and you shoud fix it.
9. The harness will not execute any tool calls if they are:
- written in xml format.
- written in unrecognizable by harness tool call json schema.
- written as \`\`\`any\n...\n\`\`\` instead of \`\`\`json\n{...}\n\`\`\` format.

[MANDATORY TOOL PROTOCOL]
To perform ANY action (running commands, reading/editing files, listing directories, analyzing images), you MUST output a tool call as a JSON block in markdown:
\`\`\`json
{
  "name": "tool_name",
  "arguments": {
    "arg_name": "arg_value"
  }
}
\`\`\`

[QUESTION YOURSELF BEFORE WORKING ON THE USER REQUEST]
1. Does the goal need to be created, updated or finished?
2. Do we need subagents?
3. Is tool execution needed? If so, which tools?`;

  /**
   * `clientPrompt`: the client brings its own system prompt and workflow (an
   * autonomous agent, not DeepSeek Harness). Its system messages are kept and
   * the Harness discussion / create_goal workflow is left out; the tool list
   * and the tool-call protocol stay.
   */
  public static format(
    messages: OpenAIMessage[] = [],
    tools: OpenAITool[] = [],
    clientPrompt: boolean = false,
  ): string {
    const sections: string[] = [];

    if (clientPrompt) {
      const system = messages
        .filter((m) => m && (m.role || "").toLowerCase() === "system")
        .map((m) => this.extractMessageContent(m.content).trim())
        .filter(Boolean);
      if (system.length > 0) {
        sections.push(`[SYSTEM INSTRUCTIONS]\n${system.join("\n\n")}`);
      }
    }

    const toolsHeader = clientPrompt ? this.formatToolList(tools) : this.formatTools(tools);
    if (toolsHeader) {
      sections.push(toolsHeader);
    }

    const conversationMessages: Array<
      OpenAIMessage & { normalizedContent: string }
    > = [];
    const toolCallIdToNameMap = new Map<string, string>();

    for (const msg of messages) {
      if (!msg || typeof msg !== "object") continue;
      const rawContent = this.extractMessageContent(msg.content);
      const content = this.stripRulesAndProtocol(rawContent);
      const role = (msg.role || "user").toLowerCase() as any;

      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          const name = tc?.function?.name;
          if (tc?.id && name) {
            toolCallIdToNameMap.set(tc.id, name);
          }
        }
      }
      if (msg.function_call?.name) {
        toolCallIdToNameMap.set("legacy_function_call", msg.function_call.name);
      }

      if (role !== "system") {
        conversationMessages.push({ ...msg, role, normalizedContent: content });
      }
    }

    if (conversationMessages.length > 0) {
      const dialogParts: string[] = [];

      for (const msg of conversationMessages) {
        const role = msg.role;
        const content = msg.normalizedContent || "";

        if (role === "user") {
          dialogParts.push(`User:\n${content}`);
        } else if (role === "assistant") {
          let text = "";
          if (msg.reasoning_content || msg.thought) {
            const thought = this.stripRulesAndProtocol(
              (msg.reasoning_content || msg.thought || "").trim(),
            );
            if (thought) {
              text += `<think>\n${thought}\n</think>\n`;
            }
          }

          if (content) {
            const cleanContentText = content.replace(/```\s*$/, "").trim();
            if (cleanContentText) {
              text += `${cleanContentText}\n`;
            }
          }

          if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
            for (const tc of msg.tool_calls) {
              const fn = tc.function || tc;
              let argsObj = fn.arguments;
              if (typeof argsObj === "string") {
                try {
                  argsObj = JSON.parse(argsObj);
                } catch {}
              }
              const formattedArgs =
                typeof argsObj === "object" && argsObj !== null
                  ? JSON.stringify(argsObj, null, 2)
                  : JSON.stringify({ raw_arguments: argsObj }, null, 2);

              text += `\`\`\`json\n{\n  "name": "${fn.name}",\n  "arguments": ${formattedArgs}\n}\n\`\`\`\n`;
            }
          }

          dialogParts.push(`Assistant:\n${text.trim()}`);
        } else if (role === "tool" || role === "function") {
          const resolvedName =
            msg.name ||
            (msg.tool_call_id
              ? toolCallIdToNameMap.get(msg.tool_call_id)
              : null) ||
            msg.tool_call_id ||
            "tool";

          dialogParts.push(
            `Harness tool execution report for "${resolvedName}": \`\`\`\n${content}\n\`\`\``,
          );
        } else {
          dialogParts.push(`[${role.toUpperCase()}]:\n${content}`);
        }
      }

      if (dialogParts.length > 0) {
        sections.push(`[CONVERSATION HISTORY]\n${dialogParts.join("\n\n")}`);
      }
    }

    sections.push(this.RULES_AND_PROTOCOL);
    return sections.join("\n\n---\n\n");
  }

  private static describeTools(tools: OpenAITool[]) {
    return tools.map((t) => {
      const fn = t.function || t;
      return {
        name: fn.name,
        description: fn.description || "",
        parameters: fn.parameters || (t as any).input_schema || {},
      };
    });
  }

  private static formatToolList(tools: OpenAITool[]): string {
    if (!tools || !Array.isArray(tools) || tools.length === 0) return "";
    return `[AVAILABLE TOOLS]
You have access to the following tools:
\`\`\`json
${JSON.stringify(this.describeTools(tools), null, 2)}
\`\`\``;
  }

  private static formatTools(tools: OpenAITool[]): string {
    if (!tools || !Array.isArray(tools) || tools.length === 0) return "";
    const toolDescriptions = this.describeTools(tools);

    return `[SYSTEM ARCHITECTURE & EXECUTION ENVIRONMENT]
CRITICAL NOTICE: You are an autonomous backend engine connected to an automated execution harness via JSON-RPC.
Too make your usage free via Harness, the user refused to pay for the API and made a special bridge between the free browser version of DeepSeek and Harness via a proxy that translates your messages to Harness.

[AVAILABLE TOOLS]
You have access to the following tools:
\`\`\`json
${JSON.stringify(toolDescriptions, null, 2)}
\`\`\`

[GENERAL KNOWLEDGE]
You are an autonomous AI coding agent with real-time access to the local environment, filesystem, and specialized tools powered by DeepSeek Harness (DSH), operating through the DeepSeek Chat Web Bridge in a browser-based environment.

[WORKFLOW]

1. DISCUSSION
- The user describes what they want to accomplish.
- Discuss the requirements and proposed approach with the user.
- Do not start executing the task yet.

2. GOAL DEFINITION
- Once the plan is agreed upon, use the \`create_goal\` tool to define the goal.
- Wait for the user to explicitly request that you start working.

3. EXECUTION LOOP
- After the user asks you to start, begin the work loop:
1. Make a tool call.
2. Wait for the Harness report.
3. Analyze the result.
4. Decide what should be done next.
5. Make the next tool call.
- Continue this loop until you believe the goal has been accomplished.

4. COMPLETION
- When you believe the goal is accomplished, stop the execution loop.
- Report what was done and present the result to the user.
- Do not close the goal yet.

5. USER REVIEW
- Wait for the user's feedback.
- If the user says they are satisfied:
→ use the appropriate tool call to close the goal
→ end the workflow.
- If the user is not satisfied or requests changes:
→ return to step 1 (DISCUSSION)
→ discuss the requested changes
→ redefine the goal using \`create_goal\`
→ wait for the user to explicitly request starting again.`;
  }

  public static stripRulesAndProtocol(content: string): string {
    if (!content || typeof content !== "string") return "";

    return (
      content
        // 1. Полный блок правил до следующей секции заголовка или конца текста
        .replace(
          /\[RULES\]:?[\s\S]*?(?=\n\s*\[(?:SYSTEM ARCHITECTURE|AVAILABLE TOOLS|GENERAL KNOWLEDGE|WORKFLOW|CONVERSATION HISTORY)\]|$)/gi,
          "",
        )
        // 2. Изолированный блок [MANDATORY TOOL PROTOCOL] с кодовым блоком
        .replace(
          /\[MANDATORY TOOL PROTOCOL\][\s\S]*?(?:```(?:json)?[\s\S]*?```|(?=\n\s*\[[A-Z\s]+\])|$)/gi,
          "",
        )
        // 3. Блок [QUESTION YOURSELF...] вместе со списком вопросов/ответов (1., 2., -, *)
        .replace(
          /\[QUESTION YOURSELF BEFORE WORKING ON THE USER REQUEST\](?:\s*\n\s*(?:\d+[\.\)]|\-|\*)[^\n]+)*/gi,
          "",
        )
        // 4. Очистка разделителей и лишних пустых строк
        .replace(/^\s*---\s*$/gm, "")
        .replace(/\n{3,}/g, "\n\n")
        .trim()
    );
  }

  private static extractMessageContent(
    content: OpenAIMessage["content"],
  ): string {
    if (content === null || content === undefined) return "";
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return (content as OpenAIContentPart[])
        .map((part) => {
          if (!part) return "";
          if (typeof part === "string") return part;
          if (part.type === "text" || part.type === "input_text")
            return part.text || "";
          if (part.type === "image_url") {
            const url =
              typeof part.image_url === "string"
                ? part.image_url
                : part.image_url?.url || "";
            return url ? `[Image URL: ${url}]` : "[Image]";
          }
          if (part.type === "image") return "[Attached Image]";
          if (part.type === "tool_result") {
            return typeof part.content === "string"
              ? part.content
              : JSON.stringify(part.content || "");
          }
          return "";
        })
        .filter(Boolean)
        .join("\n");
    }
    if (typeof content === "object") {
      try {
        return JSON.stringify(content);
      } catch {
        return "";
      }
    }
    return String(content);
  }
}

/**
 * Экспорт функции для обратной совместимости с chat.controller.ts
 */
export function formatMessagesToPrompt(
  messages: OpenAIMessage[] = [],
  tools: OpenAITool[] = [],
  clientPrompt: boolean = false,
): string {
  return PromptFormatterService.format(messages, tools, clientPrompt);
}

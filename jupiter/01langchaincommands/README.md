# langchaincommands

Java (LangChain4j) version of two notebooks: `Different Langchain Commands.ipynb` (requirements 1–2) and
`morelangchaincommands.ipynb` (more chat lessons, tool calling and RAG).

Every entry point does **one thing only**, all entry-point classes end in `Main`, and each type of work has **its
own package** (under `com.jupiter.langchaincommands`):

```
search/                     LlmWithoutTavilyMain                            1A   search WITHOUT Tavily: plain LLM
                            DeepAgentWithTavilyMain, DeepAgent,             1B   search WITH Tavily: deep agent
                            TavilySearch
chat/invoke/                SimpleInvokeMain                                M1   the simplest llm.invoke
chat/messages/              SystemHumanMessagesMain                         2.1  system + human messages
chat/chatprompttemplate/    DraftEmailTemplateMain, MedicalTemplateMain,    2.2  ChatPromptTemplate chains
                            MessagesTemplateMain, MessagesTemplate,         M2   from_messages (system + human)
                            PromptChain, Templates
chat/selfcorrectingprompt/  ReflectionMain, ReflectionChain                 2.3  draft -> critique -> refine
chat/structuredoutput/      StructuredOutputMain, CourseInfo                M3   typed output (Pydantic)
tools/                      ToolCallRequestMain, ToolCallLoopMain,          M4   tool calling
                            CourseTools
rag/                        RagPipeline                                     R    all RAG steps in one class
rag/loader/                 WebLoaderMain                                   R1   load a web page
rag/chunking/               ChunkingMain                                    R2   split into chunks
rag/embeddings/             EmbeddingsMain                                  R3   text -> vectors
rag/vectorstore/            VectorStoreSearchMain                           R4   store + similarity search
rag/ragchain/               RagChainMain                                    R5   retrieve + answer
common/                     Config, Models                                  shared settings and model set-up
```

**search** holds both answers to a fresh-facts question (without and with internet search); **chat** holds the
chat lessons, one sub-package each; **tools** and **rag** hold the tool-calling and RAG lessons. Lesson packages
only depend on earlier lessons: `chat.selfcorrectingprompt` reuses `PromptChain` and the draft template from
`chat.chatprompttemplate`, and `rag.ragchain` reuses the vector-store search from `rag.vectorstore`.

| Requirement | Entry point (main class) | Maven command | Notebook cells | Needs |
|---|---|---|---|---|
| 1A. The LLM **without** Tavily search | `search.LlmWithoutTavilyMain` | `mvn compile exec:java@without-tavily` | 0–2 | `OPENAI_API_KEY` |
| 1B. Deep agent **with** Tavily search | `search.DeepAgentWithTavilyMain` | `mvn compile exec:java@with-tavily` | 3–9 | `OPENAI_API_KEY`, `TAVILY_API_KEY` |
| 2.1 Messages with roles | `chat.messages.SystemHumanMessagesMain` | `mvn compile exec:java@messages` | 10–15 | `OPENAI_API_KEY` |
| 2.2a Prompt template – draft email | `chat.chatprompttemplate.DraftEmailTemplateMain` | `mvn compile exec:java@draft-template` | 16–20 | `OPENAI_API_KEY` |
| 2.2b Prompt template – medicine summary | `chat.chatprompttemplate.MedicalTemplateMain` | `mvn compile exec:java@medical-template` | 21–23 | `OPENAI_API_KEY` |
| 2.3 Self-correction (reflection) | `chat.selfcorrectingprompt.ReflectionMain` | `mvn compile exec:java@reflection` | 24–30 | `OPENAI_API_KEY` |

More lessons from `morelangchaincommands.ipynb` are listed in [More LangChain commands](#more-langchain-commands).

Keys are read from the environment, or from the nearest `.env` file (e.g. `jupiter/.env`).
All demos make real, paid OpenAI calls (and Tavily calls for the agent).

## Requirement 1 – Search without and with Tavily (`search` package)

*Ask the LLM a question; if it does not know the answer, it searches the internet and answers from what it finds.*

Run the two entry points one after the other and compare the answers to the same question,
"Latest News in AI Space as of Sep 16th 2026":

| Entry point | What it shows |
|---|---|
| `LlmWithoutTavilyMain` | gpt-5.4 asked directly – it can only answer from training data and says it cannot browse. |
| `DeepAgentWithTavilyMain` | A **deep agent** with a Tavily search tool: it writes a to-do plan, searches the web, saves notes to its virtual files, and writes a report with real, dated headlines. |

```
mvn compile exec:java@without-tavily     # 1A: plain LLM, no internet
mvn compile exec:java@with-tavily        # 1B: deep agent + Tavily
```

| Class | Package | Role |
|---|---|---|
| `LlmWithoutTavilyMain` | `search` | **Entry point 1A.** Explains why a plain LLM cannot answer questions about fresh facts. |
| `DeepAgentWithTavilyMain` | `search` | **Entry point 1B.** Explains the agent loop, what makes an agent "deep", and LangGraph vs Deep Agents. |
| `DeepAgent` | `search` | Java version of `create_deep_agent`: planning (`write_todos`), virtual files (`ls`, `read_file`, `write_file`, `edit_file`) and `internet_search`. |
| `TavilySearch` | `search` | Java version of `TavilyClient.search` (Tavily REST API). |

Rule of thumb: if you can draw the workflow as a flowchart in advance, use **LangGraph**; if you can only describe
the goal and want the agent to work out the steps, use **Deep Agents**.

## Requirement 2 – Chat with prompts (`chat` package: `messages`, `chatprompttemplate`, `selfcorrectingprompt`)

One entry point and one package per type of prompt work, in teaching order:

| Entry point | Type of work | What it shows |
|---|---|---|
| `SystemHumanMessagesMain` | Messages with roles | **SystemMessage** (who the model is) + **HumanMessage** (the question) → a 3-day corporate AI training plan. |
| `DraftEmailTemplateMain` | Reusable prompt template | **ChatPromptTemplate** with an `{input}` placeholder, run as an LCEL chain `prompt \| llm \| StrOutputParser()` → a polite email declining the client's low-budget offer. |
| `MedicalTemplateMain` | Reusable prompt template | The same technique for a different job: `{Name}` = "Creatine" → a 2-line pros/cons/side-effects summary with the BEPEC sign-off. |
| `ReflectionMain` | Self-correction | **Reflection**: draft → critique → refine, three chained prompts → a warmer, improved email. |

```
mvn compile exec:java@messages
mvn compile exec:java@draft-template
mvn compile exec:java@medical-template
mvn compile exec:java@reflection
```

| Class | Package | Role |
|---|---|---|
| `SystemHumanMessagesMain` | `chat.messages` | **Entry point 2.1.** Explains messages and roles. |
| `DraftEmailTemplateMain` | `chat.chatprompttemplate` | **Entry point 2.2a.** Explains prompt templates and LCEL chains with the draft email. |
| `MedicalTemplateMain` | `chat.chatprompttemplate` | **Entry point 2.2b.** The same technique for a different job (medicine summary). |
| `ReflectionMain` | `chat.selfcorrectingprompt` | **Entry point 2.3.** Explains self-correction and its cost (3 calls instead of 1). |
| `PromptChain` | `chat.chatprompttemplate` | Java version of `ChatPromptTemplate.from_template(t) \| llm \| StrOutputParser()`, with the same `{name}` placeholders. |
| `Templates` | `chat.chatprompttemplate` | The notebook's draft and medical templates and the example client offer, shared by the template and reflection lessons. |
| `ReflectionChain` | `chat.selfcorrectingprompt` | The draft → critique → refine chain (Python's `RunnablePassthrough` version) and its critique and refine templates. |

## More LangChain commands

From `morelangchaincommands.ipynb`: one entry point per example, grouped by topic.

| # | Entry point | Package | Maven command | Notebook cells | What it shows | Needs |
|---|---|---|---|---|---|---|
| M1 | `SimpleInvokeMain` | `chat.invoke` | `mvn compile exec:java@invoke` | 0–2 | The simplest call, `llm.invoke("What is AI in 1 Line")`, plus the reply's metadata (model, tokens, finish reason, id). | `OPENAI_API_KEY` |
| M2 | `MessagesTemplateMain` | `chat.chatprompttemplate` | `mvn compile exec:java@messages-template` | 3–7 | `ChatPromptTemplate.from_messages`: a system **and** a human template with variables (domain, topic, audience). | `OPENAI_API_KEY` |
| M3 | `StructuredOutputMain` | `chat.structuredoutput` | `mvn compile exec:java@structured-output` | 8–17 | **Structured output**: the model must answer as JSON matching `CourseInfo`, returned as a typed Java record. | `OPENAI_API_KEY` |
| M4a | `ToolCallRequestMain` | `tools` | `mvn compile exec:java@tool-calls` | 18–23 | **Tool calling**: which tools (`get_course_fee`, `get_batch_dates`) the model asks for, with arguments and ids. | `OPENAI_API_KEY` |
| M4b | `ToolCallLoopMain` | `tools` | `mvn compile exec:java@tool-loop` | 24–25 | The tool loop by hand: run the requested tools, send results back, get the final answer. | `OPENAI_API_KEY` |
| R1 | `WebLoaderMain` | `rag.loader` | `mvn compile exec:java@web-loader` | 26–28 | `WebBaseLoader`: a web page as a Document (text + source, title, description, language). | none |
| R2 | `ChunkingMain` | `rag.chunking` | `mvn compile exec:java@chunking` | 29–37 | `RecursiveCharacterTextSplitter`: chunks of at most 200 characters with 30 characters of overlap. | none |
| R3 | `EmbeddingsMain` | `rag.embeddings` | `mvn compile exec:java@embeddings` | 38–44 | Embeddings: text -> 1536 numbers; also shows each sentence's similarity to the question. | `OPENAI_API_KEY` |
| R4 | `VectorStoreSearchMain` | `rag.vectorstore` | `mvn compile exec:java@vector-search` | 45–51 | A vector store (FAISS -> in-memory store) and similarity search by meaning. | `OPENAI_API_KEY` |
| R5 | `RagChainMain` | `rag.ragchain` | `mvn compile exec:java@rag-chain` | 52–60 | The full RAG chain: retrieve 2 facts, answer only from them. | `OPENAI_API_KEY` |

All five RAG lessons call **`rag.RagPipeline`**, one class with every RAG step in pipeline order: `loadWebPage` /
`fromHtml` (load), `split` / `splitDocument` (chunk), `embedQuery` / `embedDocuments` / `cosine` (embed),
`fromDocuments` / `fromSegments` / `similaritySearch` (store and search) and `formatDocs` / `answer` (the RAG chain).
Use it directly to build your own RAG flow, e.g. load a page -> split -> store -> answer.

Supporting classes: `MessagesTemplate` (a two-message template), `CourseInfo` (the record the model fills in, like
the Pydantic model) and `CourseTools` (the two `@tool` functions, plus running a tool call by name).

Differences from Python, explained in the class comments: FAISS is replaced by LangChain4j's in-memory store, the
JDK's built-in HTML parser replaces BeautifulSoup (no extra library), and LangChain4j's recursive splitter has a fixed separator order, so `ChunkingMain` gives 15
chunks where the notebook got 13.

## Shared code (`common` package)

| Class | Role |
|---|---|
| `Config` | Reads `OPENAI_API_KEY` / `TAVILY_API_KEY` from the environment or the nearest `.env`. |
| `Models` | Builds OpenAI chat models (normal, and strict JSON-schema for structured output) and embedding models, sets up the console (UTF-8) and prints section headers. |

## Running from IntelliJ

Open the entry point you want (any class ending in `Main`) and run its `main`. No program arguments are needed.

## Tests

```
mvn test
```

Offline tests (`src/test`, next to the code they test) use fake models from `testutil.Fakes` (word-count
embeddings, scripted chat replies), so they need no API keys and cost nothing. They cover the templates, the
reflection chain, structured output, the tools and tool loop, the web loader (on sample HTML), chunking, vector
search and the RAG chain (`rag.RagPipelineTest` tests every pipeline step). For example:
`chat.chatprompttemplate.PromptChainTest` checks that templates fill their placeholders, and
`chat.selfcorrectingprompt.ReflectionChainTest` checks that the reflection chain passes the draft and critique along.

package com.bepec.ragpinecone;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pinecone.PineconeEmbeddingStore;
import dev.langchain4j.store.embedding.pinecone.PineconeServerlessIndexConfig;
import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Settings, API keys and the three external services: OpenAI chat model, OpenAI embedding model and the Pinecone
 * vector store. The Java version of the notebooks' {@code load_dotenv()}, {@code Pinecone(...)},
 * {@code pc.create_index(...)}, {@code OpenAIEmbeddings(...)}, {@code PineconeVectorStore(...)} and
 * {@code ChatOpenAI(...)}.
 */
public final class Config {

    /** Same names and settings as the notebooks, so Java and Python share one Pinecone index. */
    public static final String INDEX_NAME = "langchain-demo";
    public static final String PDF_NAMESPACE = "pdf-docs";
    /** "" is Pinecone's default namespace - what rag.ipynb uses when it passes no namespace. */
    public static final String DEFAULT_NAMESPACE = "";
    public static final int DIMENSION = 1536;
    public static final String CLOUD = "aws";
    public static final String REGION = "us-east-1";
    /**
     * Metadata key that holds each chunk's text. LangChain (Python) stores it under "text"; LangChain4j's default is
     * "text_segment". Using "text" means vectors written by the notebooks and by this Java code can be read by both.
     */
    public static final String TEXT_KEY = "text";

    public static final String DEFAULT_PDF = "AI Agenda_Updated'26.pdf";
    public static final String CHAT_MODEL = "gpt-4o-mini";
    public static final String EMBEDDING_MODEL = "text-embedding-3-small";

    private static final Dotenv DOTENV = loadDotenv();

    private Config() {
    }

    /** Looks for .env in the working directory and its parents, then upward from the compiled classes. */
    private static Dotenv loadDotenv() {
        Path dir = findEnvDir(Path.of("").toAbsolutePath());
        if (dir == null) {
            dir = findEnvDir(codeLocation());
        }
        var cfg = Dotenv.configure().ignoreIfMissing();
        if (dir != null) {
            cfg = cfg.directory(dir.toString());
        }
        return cfg.load();
    }

    private static Path findEnvDir(Path start) {
        Path dir = start;
        while (dir != null && !Files.exists(dir.resolve(".env"))) {
            dir = dir.getParent();
        }
        return dir;
    }

    private static Path codeLocation() {
        try {
            return Path.of(Config.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        } catch (Exception e) {
            return null;
        }
    }

    /** Environment variables win over values from .env. */
    public static String env(String name) {
        String value = System.getenv(name);
        return value != null && !value.isBlank() ? value : DOTENV.get(name);
    }

    /** Reads a required key (OPENAI_API_KEY, PINECONE_API_KEY) and fails with a clear message if it is missing. */
    static String required(String name) {
        String value = env(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException(name + " not found. Add it to your .env file: " + name + "=your_key_here");
        }
        return value;
    }

    /** The answering model: {@code ChatOpenAI(model="gpt-4o-mini", temperature=0)} - repeatable, factual answers. */
    public static ChatModel llm() {
        return OpenAiChatModel.builder()
                .apiKey(required("OPENAI_API_KEY"))
                .modelName(CHAT_MODEL)
                .temperature(0.0)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /**
     * The embedding model: {@code OpenAIEmbeddings(model="text-embedding-3-small")}. Produces vectors of 1536
     * numbers - which is why the Pinecone index is created with dimension 1536. If you change the model, the index
     * dimension must change too.
     */
    public static EmbeddingModel embeddings() {
        return OpenAiEmbeddingModel.builder()
                .apiKey(required("OPENAI_API_KEY"))
                .modelName(EMBEDDING_MODEL)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /**
     * Connects to the Pinecone index and returns a vector store for one namespace - the Java version of:
     * <pre>
     * pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
     * if not pc.has_index("langchain-demo"):
     *     pc.create_index(name="langchain-demo", dimension=1536, metric="cosine",
     *                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
     * vector_store = PineconeVectorStore(index=pc.Index("langchain-demo"), embedding=embeddings, namespace=...)
     * </pre>
     * <ul>
     *   <li><b>Index</b> - a Pinecone database of vectors. Created on the first run only (serverless on AWS
     *       us-east-1, dimension 1536, cosine similarity); afterwards the existing index is reused, and the data in
     *       it stays there between runs - unlike the in-memory store of the plain ragchatbot project.</li>
     *   <li><b>Namespace</b> - a separate partition inside the index. The PDF chunks live in "pdf-docs", the basics
     *       demo uses the default namespace, so they never mix in searches.</li>
     * </ul>
     * Creating a new index takes a little while and counts towards your Pinecone plan's index limit.
     *
     * @param namespace partition to read and write, e.g. {@link #PDF_NAMESPACE} or {@link #DEFAULT_NAMESPACE}
     */
    public static EmbeddingStore<TextSegment> pinecone(String namespace) {
        return PineconeEmbeddingStore.builder()
                .apiKey(required("PINECONE_API_KEY"))
                .index(INDEX_NAME)
                .nameSpace(namespace)
                .metadataTextKey(TEXT_KEY)
                .createIndex(PineconeServerlessIndexConfig.builder()
                        .dimension(DIMENSION)
                        .cloud(CLOUD)
                        .region(REGION)
                        .build())
                .build();
    }
}

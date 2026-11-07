package com.bepec.ragpinecone;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * BEPEC PDF RAG Chatbot with Pinecone - main class. Java port of pdf_rag.ipynb.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java                                    index "AI Agenda_Updated'26.pdf", then ask questions
 *   mvn compile exec:java -Dexec.args="file.pdf"             index another PDF, then ask questions
 *   mvn compile exec:java -Dexec.args="file.pdf What is covered in Python"   one question, then exit
 *   mvn compile exec:java -Dexec.args="--skip-index"         ask questions about what is already in Pinecone
 * </pre>
 * Needs OPENAI_API_KEY and PINECONE_API_KEY (environment or .env). The PDF path can also come from PDF_PATH.
 *
 * <h2>Key idea: RAG with a persistent vector database</h2>
 * RAG (Retrieval-Augmented Generation): find the few passages of the PDF that are relevant to a question and give
 * only those to the model, which answers from them and cites the pages. The plain ragchatbot project keeps its
 * vectors in memory and rebuilds them on every start. Here they live in <b>Pinecone</b>, a managed cloud vector
 * database: once indexed, the PDF stays searchable across runs, from other programs, and from the Python notebook
 * (same index, namespace, ids and text key) - so indexing is a one-time cost and {@code --skip-index} can go
 * straight to questions.
 *
 * <h2>The pipeline</h2>
 * <pre>
 *  INDEXING (skipped with --skip-index)
 *    PDF --PdfLoader--> pages --split(1000, overlap 150)--> chunks --MD5(source::text)--> stable ids
 *        --OpenAI embeddings, 100 per batch--> vectors --upsert--> Pinecone index "langchain-demo",
 *                                                                  namespace "pdf-docs"
 *  ANSWERING (every question)
 *    question --embedding--> vector --Pinecone similarity search--> top 3 chunks
 *        --"[Page n]" labels--> prompt (rules + context + question) --gpt-4o-mini, temp 0--> answer + pages
 * </pre>
 *
 * <h2>Pinecone terms</h2>
 * <pre>
 * | Term      | Here                                                                         |
 * |-----------|------------------------------------------------------------------------------|
 * | Index     | "langchain-demo": serverless (AWS us-east-1), 1536 dimensions, cosine metric |
 * |           | - created on the first run if it does not exist                              |
 * | Namespace | "pdf-docs": a partition inside the index; searches only see this namespace   |
 * | Vector    | one chunk: id (MD5) + 1536 numbers + metadata (text, page, source, ...)      |
 * | Upsert    | insert, or overwrite when the id already exists - so re-indexing is safe     |
 * </pre>
 *
 * <h2>Notebook cell -> Java</h2>
 * <pre>
 * | pdf_rag.ipynb                                       | Java                                     |
 * |-----------------------------------------------------|------------------------------------------|
 * | load_dotenv(), PDF_PATH, NAMESPACE                  | Config                                   |
 * | PyPDFLoader(PDF_PATH).load()                        | PdfLoader.load                           |
 * | Pinecone(...), has_index / create_index, Index(...) | Config.pinecone(namespace)               |
 * | OpenAIEmbeddings, PineconeVectorStore(...)          | Config.embeddings(), Config.pinecone     |
 * | RecursiveCharacterTextSplitter(1000, 150)           | PineconeRagPipeline.split                |
 * | ids = [md5(f"{source}::{text}") ...]                | PineconeRagPipeline.chunkId              |
 * | add_documents(chunks, ids=ids, batch_size=100)      | PineconeRagPipeline.index                |
 * | as_retriever(..., k=3)                              | PineconeRagPipeline.retrieve             |
 * | ChatOpenAI("gpt-4o-mini", temperature=0), prompt    | Config.llm(), PineconeRagPipeline.buildPrompt |
 * | format_docs, rag_chain.invoke(question)             | PineconeRagPipeline.formatDocs / answer  |
 * </pre>
 *
 * <h2>Limits</h2>
 * Each question is answered on its own (no chat memory). Re-indexing adds or overwrites chunks but never deletes:
 * if a PDF changes, old chunks stay in the namespace until you remove them (e.g. in the Pinecone console).
 * Pinecone and OpenAI calls use your accounts' quotas.
 *
 * @see PineconeBasicsMain the Java port of rag.ipynb (Pinecone basics: add, search, filter)
 */
public final class PdfRagChatbotMain {

    static final String SKIP_INDEX = "--skip-index";

    private PdfRagChatbotMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li><b>Read the arguments</b> - an optional {@code --skip-index} flag, an optional PDF path (else PDF_PATH,
     *       else {@value Config#DEFAULT_PDF}), and optional question words.</li>
     *   <li><b>Connect</b> - build the OpenAI embedding and chat models and the Pinecone store for namespace
     *       "pdf-docs" ({@link Config#pinecone}), creating the index on the very first run. Missing API keys fail
     *       here with a clear message.</li>
     *   <li><b>Index</b> (unless {@code --skip-index}) - load the PDF page by page, split, embed and upsert into
     *       Pinecone ({@link PineconeRagPipeline#index}). Prints "Loaded N pages", "Split into N chunks" and
     *       "Stored in Pinecone", like the notebook.</li>
     *   <li><b>Answer</b> - with question words: answer that one question and exit. Without: an interactive loop
     *       ({@link #chatLoop}).</li>
     * </ol>
     *
     * @param args optional: [--skip-index] [pdf path] [question words...]
     */
    public static void main(String[] args) throws IOException {
        List<String> rest = new ArrayList<>(Arrays.asList(args));
        boolean skipIndex = rest.remove(SKIP_INDEX);

        var rag = new PineconeRagPipeline(Config.embeddings(), Config.pinecone(Config.PDF_NAMESPACE), Config.llm());

        if (!skipIndex) {
            String pdfSetting = !rest.isEmpty() ? rest.removeFirst() : Config.env("PDF_PATH");
            Path pdf = Config.resolveFile(pdfSetting == null || pdfSetting.isBlank() ? Config.DEFAULT_PDF : pdfSetting);

            var pages = PdfLoader.load(pdf);
            System.out.println("Loaded " + pages.size() + " pages from " + pdf.getFileName());
            int chunks = rag.index(pages);
            System.out.println("Split into " + chunks + " chunks");
            System.out.println("Stored in Pinecone (index '" + Config.INDEX_NAME + "', namespace '"
                    + Config.PDF_NAMESPACE + "')");
        }

        if (!rest.isEmpty()) {
            String question = String.join(" ", rest);
            System.out.println("\nQ: " + question);
            System.out.println(rag.answer(question));
            return;
        }
        chatLoop(rag);
    }

    /**
     * Interactive loop: one question per line, answered with the full retrieve -> prompt -> answer chain. An empty
     * line, "exit", "quit" or end of input stops. If one answer fails (network, quota), the error is shown and the
     * loop continues.
     */
    static void chatLoop(PineconeRagPipeline rag) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        System.out.println("\nAsk a question about the PDF (empty line or 'exit' to quit).");
        while (true) {
            System.out.print("\nYou: ");
            String question = in.readLine();
            if (question == null || question.isBlank()
                    || question.strip().equalsIgnoreCase("exit") || question.strip().equalsIgnoreCase("quit")) {
                return;
            }
            try {
                System.out.println("Bot: " + rag.answer(question.strip()));
            } catch (RuntimeException e) {
                System.out.println("Bot: Sorry, something went wrong: " + e.getMessage());
            }
        }
    }
}

package com.bepec.ragchatbot;

import dev.langchain4j.data.document.Document;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * BEPEC RAG Chatbot - main class. Java port of ragchatbot.ipynb: ask questions about a PDF and get answers based
 * only on that PDF, with page numbers.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java                                         (uses "AI Agenda_Updated'26.pdf" in this folder)
 *   mvn compile exec:java -Dexec.args="path/to/file.pdf"          (another PDF, then ask questions interactively)
 *   mvn compile exec:java -Dexec.args="file.pdf What they cover under python"   (one question, then exit)
 * </pre>
 * Needs OPENAI_API_KEY in the environment or a .env file. The PDF path can also be set with PDF_PATH.
 *
 * <h2>Key idea: RAG = Retrieval-Augmented Generation</h2>
 * The model has never seen your PDF, and the whole PDF is too long (and too expensive) to send with every question.
 * So we <b>retrieve</b> only the few passages relevant to the question and <b>augment</b> the prompt with them;
 * the model then <b>generates</b> an answer from those passages only. This keeps answers grounded in the document,
 * lets it cite pages, and lets it say "I don't know" instead of inventing things.
 *
 * <h2>The pipeline</h2>
 * <pre>
 *  INDEXING (once, at start-up)
 *    PDF --PdfLoader--> 35 pages --splitter(1000, overlap 150)--> ~50 chunks
 *        --embedding model--> 50 vectors (1536 numbers each) --> in-memory vector store
 *
 *  ANSWERING (every question)
 *    question --embedding model--> vector --similarity search--> top 4 chunks
 *        --format_docs--> "[Page 5] ... [Page 7] ..."  --> prompt (rules + context + question)
 *        --gpt-4o-mini (temperature 0)--> answer with page numbers
 * </pre>
 *
 * <h2>Notebook cell -> Java</h2>
 * <pre>
 * | Notebook                                         | Java                                        |
 * |--------------------------------------------------|---------------------------------------------|
 * | load_dotenv()                                    | Config (reads .env)                         |
 * | PyPDFLoader(PDF_PATH).load(), len(pages)         | PdfLoader.load(pdf), pages.size()           |
 * | RecursiveCharacterTextSplitter(1000, 150)        | RagPipeline.index - DocumentSplitters       |
 * | OpenAIEmbeddings("text-embedding-3-small")       | Config.embeddings()                         |
 * | Chroma.from_documents(..., "pdf_docs")           | RagPipeline.index - InMemoryEmbeddingStore  |
 * | vector_store.get(...) / vector.shape             | printed "Vector size: 1536"                 |
 * | as_retriever(search_type="similarity", k=4)      | RagPipeline.retrieve                        |
 * | ChatOpenAI("gpt-4o-mini", temperature=0)         | Config.llm()                                |
 * | ChatPromptTemplate (system + human)              | RagPipeline.buildPrompt                     |
 * | format_docs(docs)                                | RagPipeline.formatDocs                      |
 * | rag_chain.invoke(question)                       | RagPipeline.answer                          |
 * </pre>
 *
 * <h2>Example</h2>
 * Question "What they cover under python" -> the 4 chunks about the Python module are retrieved -> the answer lists
 * the topics (Introduction to Python, data types, lists/tuples/sets/dictionaries, loops, functions, OOPS, ...) and
 * names the page(s) they came from.
 *
 * <h2>Limits</h2>
 * The index lives in memory and is rebuilt (and the chunks re-embedded, a small OpenAI cost) on every start.
 * Each question is answered on its own - there is no chat memory between questions. PDFs that are scanned images
 * without a text layer would need OCR first.
 */
public final class RagChatbotMain {

    private RagChatbotMain() {
    }

    /**
     * Entry point - builds the index for one PDF, then answers questions about it.
     * <ol>
     *   <li><b>Choose the PDF</b> - the first argument, else the PDF_PATH setting, else
     *       {@value Config#DEFAULT_PDF} in the working directory.</li>
     *   <li><b>Create the models</b> ({@link Config#embeddings()}, {@link Config#llm()}) - fails right away with a
     *       clear message if OPENAI_API_KEY is missing.</li>
     *   <li><b>Load the pages</b> ({@link PdfLoader#load}) and print the page count, like {@code len(pages)}.</li>
     *   <li><b>Index</b> ({@link RagPipeline#index}) - split, embed and store; prints the chunk count and the vector
     *       size, like {@code len(chunks)} and {@code vector.shape} in the notebook.</li>
     *   <li><b>Answer</b> - if more arguments follow the PDF path, they form one question: answer it and exit.
     *       Otherwise start an interactive loop: type a question, get an answer; an empty line or "exit" quits.</li>
     * </ol>
     *
     * @param args optional: [pdf path] [question words...]
     */
    public static void main(String[] args) throws IOException {
        String pdfSetting = args.length > 0 ? args[0] : Config.env("PDF_PATH");
        Path pdf = Config.resolveFile(pdfSetting == null || pdfSetting.isBlank() ? Config.DEFAULT_PDF : pdfSetting);

        RagPipeline rag = new RagPipeline(Config.embeddings(), Config.llm());

        List<Document> pages = PdfLoader.load(pdf);
        System.out.println("Loaded " + pages.size() + " pages from " + pdf.getFileName());

        int chunkCount = rag.index(pages);
        System.out.println("Indexed " + chunkCount + " chunks (vector size: " + rag.dimension() + ")");

        if (args.length > 1) {
            String question = String.join(" ", Arrays.copyOfRange(args, 1, args.length));
            System.out.println("\nQ: " + question);
            System.out.println(rag.answer(question));
            return;
        }

        chatLoop(rag);
    }

    /**
     * Interactive question loop: reads one question per line from the console and prints the answer. Each question
     * runs the full retrieve -> prompt -> answer chain on its own. An empty line, "exit" or "quit" (or end of input)
     * stops the program. If one answer fails (e.g. a network error), the error is shown and the loop continues.
     */
    static void chatLoop(RagPipeline rag) throws IOException {
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

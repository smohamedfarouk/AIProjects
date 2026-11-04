package com.jupiter.langchaincommands.rag.loader;

import com.jupiter.langchaincommands.common.Models;
import com.jupiter.langchaincommands.rag.RagPipeline;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;

/**
 * More LangChain commands, RAG lesson 1 - loading a web page as a Document (morelangchaincommands.ipynb cells 26-28).
 * <b>Entry point.</b>
 * <p>
 * The RAG lessons, one entry point each: this class (load) -> {@code rag.chunking.ChunkingMain} (split) ->
 * {@code rag.embeddings.EmbeddingsMain} (embed) -> {@code rag.vectorstore.VectorStoreSearchMain} (store and search)
 * -> {@code rag.ragchain.RagChainMain} (answer from retrieved context).
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@web-loader
 * </pre>
 * In IntelliJ, run this class's {@code main}. No API key needed - it only downloads one public web page.
 *
 * <h2>What a loader does</h2>
 * Every RAG pipeline starts by turning a source (a web page, a PDF, a text file...) into {@link Document}s: plain
 * text plus <b>metadata</b> describing where it came from. The text is what gets searched later; the metadata lets
 * you cite the source. {@code WebBaseLoader} does this for a URL:
 * <ol>
 *   <li>download the HTML;</li>
 *   <li>strip the tags, keeping only the visible text (Python uses BeautifulSoup, Java the JDK's built-in HTML parser);</li>
 *   <li>record source (the URL), title, description and language as metadata.</li>
 * </ol>
 *
 * <h2>Notebook result</h2>
 * <pre>
 * Loaded 1 document(s)
 * First 200 chars: LangChain overview - Docs by LangChainDocumentation Index...
 * Metadata: {'source': 'https://python.langchain.com/docs/introduction/', 'title': 'LangChain overview - Docs by
 *            LangChain', 'description': 'LangChain provides create_agent: ...', 'language': 'en'}
 * </pre>
 * The live page may have changed since, so the exact text can differ.
 */
public final class WebLoaderMain {

    static final String URL = "https://python.langchain.com/docs/introduction/";

    private WebLoaderMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 ({@link Models#initConsole()}) - no API key is needed here.</li>
     *   <li>Load the LangChain introduction page ({@link RagPipeline#loadWebPage}).</li>
     *   <li>Print how many documents were loaded, the first 200 characters and the metadata, like the notebook.</li>
     * </ol>
     *
     * @param args optional: another URL to load
     */
    public static void main(String[] args) throws Exception {
        Models.initConsole();
        String url = args.length > 0 ? args[0] : URL;
        Models.section("WebBaseLoader(\"" + url + "\").load()");
        Document doc = RagPipeline.loadWebPage(url);
        System.out.println("Loaded 1 document(s)");
        System.out.println("First 200 chars: " + doc.text().substring(0, Math.min(200, doc.text().length())));
        System.out.println("Metadata: " + doc.metadata().toMap());
    }
}

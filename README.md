# Supplemental Resources for RAG in Libraries 

**Companion to LYRASIS Presentation: "Giving AI Access to Library Knowledge"**
**David Meincke, MSLS | Johnson Wales University | November 2025**

---

## About This Document

This is your take-home resource guide for exploring RAG (Retrieval-Augmented Generation) in library contexts. Resources are organized by type and include both free/open-source and commercial options across the full spectrum from no-code tools to production frameworks.

**Key:**
- 🆓 Free/Open Source
- 💰 Paid/Commercial
- 📚 Library-Specific
- 🎓 Academic/Peer-Reviewed
- ⚡ Quick Start (try today)

---

## 1. Try This Week (No-Code Tools)

Start here if you want to experiment immediately without writing code.

### General RAG Tools

- **🆓⚡ NotebookLM** (Google)
  https://notebooklm.google.com/
  Upload documents, ask questions, get cited answers. Great for understanding RAG UX patterns. **Free tier remains (50 sources/notebook); paid tiers added in 2025.**

- **💰⚡ ChatGPT** (OpenAI)
  https://chat.openai.com/
  Upload PDFs in chat, watch it retrieve and cite. **Free tier now supports PDF upload (3 per day) - recent 2025 update.**

- **💰⚡ Claude Projects** (Anthropic)
  https://claude.ai/
  Add documents as project knowledge base. Free tier available.

- **🆓⚡ Perplexity AI**
  https://www.perplexity.ai/
  Web search + RAG with source cards. Good example of citation UX.

### Custom Chatbot Builders (No-Code)

- **💰 CustomGPT.ai**
  https://customgpt.ai/
  Build chatbots from your content without coding. Library use cases documented.

- **💰 Chatbase**
  https://www.chatbase.co/
  Train chatbots on your documents. Pricing starts free tier.

- **💰 Voiceflow**
  https://www.voiceflow.com/
  Conversational AI platform with RAG support. Visual workflow builder.

---

## 2. Library RAG Systems in Production

Real-world implementations from peer institutions and vendors.

### Academic Library Chatbots

- **📚🆓 KingbotGPT** (San José State University)
  https://library.sjsu.edu/kingbot
  **Live demo:** https://libapps.sjsu.edu/kingbot/
  **Tech stack:** LlamaIndex, GPT-4o Mini, Chroma, Streamlit
  **Blog post:** "Library-Led AI: Building a Library Chatbot" (ALA 2025)
  https://www.ala.org/sites/default/files/2025-03/Library-LedAI.pdf

- **📚 T-Rex Chatbot** (University of Calgary)
  https://library.ucalgary.ca/
  **Article:** "Implementing an AI Reference Chatbot at University of Calgary Library" (OCLC Research, Nov 2024)
  https://hangingtogether.org/implementing-an-ai-reference-chatbot-at-the-university-of-calgary-library/

- **📚 Columbia CLIO AI Enhancement** (Columbia University)
  **Announcement:** "Augmenting Discovery with AI" (May 2024)
  https://etc.cuit.columbia.edu/news/AICoP-library-augment-discovery-with-AI
  Pilot project using document vectorization for discovery enhancement.

### Discovery Layers with RAG

- **📚💰 Ex Libris Primo Research Assistant** (Clarivate)
  **Launch:** September 2024
  **Press release:** https://clarivate.com/news/clarivate-launches-generative-ai-powered-primo-research-assistant/
  RAG grounded in 5+ billion CDI records.

- **📚💰 ProQuest Summon Research Assistant** (Clarivate)
  **Launch:** March 2025
  **Announcement:** https://librarytechnology.org/pr/31135
  Based on Ex Libris RAG architecture.

- **📚💰 EBSCO AI Insights**
  **Launch:** March 2025
  **Press release:** https://www.ebsco.com/news-center/press-releases/ebsco-information-services-launches-newest-artificial-intelligence-features/
  RAG grounded on full-text articles.

### Database Platforms with RAG

- **📚💰 JSTOR AI Research Tool**
  **Beta:** August 2023 (ongoing)
  **Blog post:** "AI and Other Advanced Technologies on JSTOR: What We're Learning" (June 2024)
  https://about.jstor.org/blog/ai-and-other-advanced-technologies-on-jstor-what-were-learning/

- **📚💰 Statista Research AI**
  https://www.statista.com/research-ai/
  Cohere embeddings + Claude 3 Sonnet for statistics research.

- **📚💰 ProQuest Research Assistant**
  **Blog post:** "Academic AI-Powered Research Assistance Now Available" (Feb 2025)
  https://about.proquest.com/en/blog/2025/academic-ai-powered-research-assistance-now-available-in-proquest-central/

### Academic Research on Library RAG

- **🎓 "Prospects of RAG for Academic Library Search and Retrieval"**
  Bevara, R.V.K., et al. (2025). *Information Technology and Libraries*, 44(2).
  https://doi.org/10.5860/ital.v44i2.17361

---

## 3. RAG Frameworks & Orchestration (Low-Code to Full-Code)

Tools for building custom RAG systems with varying levels of complexity.

### Python Frameworks (Most Popular)

- **🆓 LlamaIndex**
  https://www.llamaindex.ai/
  **Docs:** https://docs.llamaindex.ai/
  Purpose-built for RAG. Excellent documentation, active community. Used by SJSU KingbotGPT.

- **🆓 LangChain**
  https://www.langchain.com/
  **Docs:** https://python.langchain.com/docs/
  General LLM orchestration framework with strong RAG support. Large ecosystem.

- **🆓 Haystack** (deepset)
  https://haystack.deepset.ai/
  Production-ready RAG framework. Good for NLP pipelines.

### Advanced RAG Architectures (2024)

- **🆓 Microsoft GraphRAG** (July 2024, v1.0 October 2024)
  **Main:** https://www.microsoft.com/en-us/research/project/graphrag/
  **GitHub:** https://github.com/microsoft/graphrag
  **Blog:** https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/
  Game-changing approach using knowledge graphs. 70-80% improvement over naive RAG. Exceptional for special collections, archives, and complex document collections. Uses LLMs to create knowledge graphs from documents.

- **🆓 AutoRAG: Automated Pipeline Optimization** (October 2024)
  **Paper:** https://arxiv.org/abs/2410.20878
  **GitHub:** https://github.com/Marker-Inc-Korea/AutoRAG
  AutoML-style automation simplifies RAG development for libraries without ML expertise. Systematically tests configurations to find optimal pipeline.

### Low-Code Platforms

- **💰 Streamlit**
  https://streamlit.io/
  **Free tier:** Yes
  Build interactive web apps for RAG demos. Fast prototyping. Used by KingbotGPT.

- **💰 Gradio**
  https://www.gradio.app/
  **Free tier:** Yes
  Create ML/AI interfaces quickly. Good for demos and MVPs.

### Deployment Platforms

- **🆓 Google Colab**
  https://colab.research.google.com/
  Free Jupyter notebooks in browser. Great for learning and sharing demos.

- **🆓 Jupyter**
  https://jupyter.org/
  Open-source notebooks for interactive code. Industry standard for data science.

- **💰 Hugging Face Spaces**
  https://huggingface.co/spaces
  **Free tier:** Yes
  Deploy Gradio/Streamlit apps. Community platform for ML models.

---

## 4. Vector Databases

Semantic search engines for RAG retrieval.

### Free/Open Source

- **🆓 ChromaDB**
  https://www.trychroma.com/
  Embedded vector database. Great for prototypes and small-scale. Used in presentation demo.

- **🆓 Weaviate**
  https://weaviate.io/
  **Free cloud tier:** Yes
  Open-source with cloud option. Production-ready.

- **🆓 Qdrant**
  https://qdrant.tech/
  **Free cloud tier:** Yes
  High-performance, Rust-based. Good for production.

- **🆓 FAISS** (Meta AI)
  https://github.com/facebookresearch/faiss
  Library for efficient similarity search. Lower-level, more control.

- **🆓 PostgreSQL pgvector** (Major 2024 development)
  **Tutorial:** https://www.tigerdata.com/blog/postgresql-as-a-vector-database-using-pgvector
  **Implementation:** https://www.enterprisedb.com/blog/rag-app-postgres-and-pgvector
  **Major advantage for libraries already using PostgreSQL** for ILS/discovery systems. Reduces infrastructure complexity by consolidating relational data and vector embeddings in one system.

### Commercial/Managed

- **💰 Pinecone**
  https://www.pinecone.io/
  **Free tier:** Yes (100k vectors)
  Fully managed, highly scalable. Popular for production.

- **💰 Elasticsearch (Vector Search)**
  https://www.elastic.co/elasticsearch/vector-database
  Enterprise search with vector capabilities.

### Learning Resources

- **Understanding Vector Databases** (Pinecone)
  https://www.pinecone.io/learn/vector-database/
  Clear explanations with diagrams.

---

## 5. Large Language Models & APIs

The generation component of RAG.

### Commercial APIs (Pay-as-you-go)

- **💰 OpenAI API** (GPT-4, GPT-4o-mini)
  https://platform.openai.com/
  **Pricing:** https://openai.com/api/pricing/
  Industry standard. GPT-4o-mini recommended for cost-effective RAG.

- **💰 Anthropic API** (Claude)
  https://www.anthropic.com/api
  **Docs:** https://docs.anthropic.com/
  Strong context windows (200k+ tokens), good for long documents.

- **💰 Google Gemini API**
  https://ai.google.dev/
  Competitive pricing, multimodal capabilities.

- **💰 Cohere**
  https://cohere.com/
  Built for enterprise, strong embeddings. Used by Statista.

### Open-Source/Local Models

- **🆓 Ollama**
  https://ollama.ai/
  Run Llama, Mistral, and other models locally. Privacy-focused.

- **🆓 Llama 3.2 & 3.3** (Meta AI, Sept-Dec 2024)
  **Release:** https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
  **RAG Tutorial:** https://zilliz.com/blog/learn-llama-3-2-and-how-to-build-rag-with-llama-with-milvus
  Enables on-premise/privacy-preserving RAG for libraries. **Vision models (11B, 90B)** for multimodal RAG, **lightweight text models (1B, 3B)** can run on standard library hardware.

- **🆓 Mistral / Mixtral** (Mistral AI, 2024)
  **Mixtral 8X22B:** https://www.datacamp.com/tutorial/mixtral-8x22b
  **RAG Tutorial:** https://towardsdatascience.com/building-evaluating-and-tracking-a-local-advanced-rag-system-mistral-7b-llamaindex-w-b-5c9c69059f92
  Cost-efficient open-source with excellent multilingual support (English, French, Italian, German, Spanish). Sparse Mixture of Experts: 141B parameters, uses only 39B actively.

- **🆓 Phi-2/Phi-3** (Microsoft)
  https://huggingface.co/microsoft/phi-2
  Small language models (2.7B-14B params) that run on modest hardware.

### Model Hosting Platforms

- **🆓 Hugging Face**
  https://huggingface.co/
  Repository of open models. Inference API available.

- **💰 Replicate**
  https://replicate.com/
  Run open models via API. Pay per use.

---

## 6. Embeddings Models

Convert text to vectors for semantic search.

### Pre-trained Models (Free)

- **🆓 Sentence Transformers**
  https://www.sbert.net/
  Open-source library. Many pre-trained models for different domains. Used in presentation demo.

- **🆓 OpenAI Embeddings**
  https://platform.openai.com/docs/guides/embeddings
  **text-embedding-3-small:** $0.02 per 1M tokens (very affordable)

- **🆓 Cohere Embeddings**
  https://cohere.com/embeddings
  Free tier available. Multilingual support.

### Model Selection Guide

- **"Massive Text Embedding Benchmark" (MTEB)**
  https://huggingface.co/spaces/mteb/leaderboard
  Leaderboard comparing embedding models across tasks.

---

## 7. Reranking Models (Critical for Production RAG)

**Why reranking matters:** Adding a reranker after initial retrieval improves accuracy by 13-25%. Rerankers process query + document together for more accurate relevance scoring than embeddings alone.

### Commercial Rerankers

- **💰 Cohere Rerank v3.5**
  https://cohere.com/rerank
  **Docs:** https://docs.cohere.com/changelog/rerank-v3.5
  State-of-the-art reranker (Dec 2024). 13.89% accuracy improvement over OpenAI embeddings alone. 100+ languages, 4096 token context.

### Open-Source Rerankers

- **🆓 Cross-Encoders** (Sentence Transformers)
  https://www.sbert.net/examples/applications/cross-encoder/README.html
  Open-source alternative. Process query + document jointly for accurate ranking.

- **🆓 BGE Reranker** (BAAI)
  https://huggingface.co/BAAI/bge-reranker-v2-m3
  Multilingual reranker from Beijing Academy of AI. Free, performant.

### Learning Resources

- **"The Missing Component in Your RAG Chatbot"** (Coalfire)
  https://coalfire.com/the-coalfire-blog/one-component-you-desperately-need-in-your-rag-chatbot-toolchain
  Explains why reranking is essential for production systems.

- **NVIDIA RAG Control Points**
  Identifies reranking as one of 15 critical RAG control points for optimization.

---

## 8. Learning Resources

Tutorials, courses, and blogs for RAG implementation.

### Blogs & News (Highly Recommended)

- **⚡ Simon Willison's Weblog**
  https://simonwillison.net/
  Daily RAG experiments, practical tools, critical analysis. Mentioned in presentation.

- **LlamaIndex Blog**
  https://www.llamaindex.ai/blog
  RAG techniques, case studies, tutorials.

- **Pinecone Learning Center**
  https://www.pinecone.io/learn/
  Vector database concepts, RAG patterns.

- **Hugging Face Blog**
  https://huggingface.co/blog
  Open-source models, RAG implementations.

### Tutorials & Courses

- **🆓 "Building RAG Applications" (LangChain)**
  https://python.langchain.com/docs/tutorials/rag/
  Step-by-step tutorial with code.

- **🆓 "RAG Tutorial" (LlamaIndex)**
  https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html
  Quickstart with clear examples.

- **🆓 "Vector Databases for Beginners"** (Weaviate)
  https://weaviate.io/developers/weaviate
  Concepts + hands-on exercises.

- **💰 "LangChain for LLM Application Development"** (DeepLearning.AI)
  https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/
  Free short course by Andrew Ng.

### YouTube Channels

- **Sam Witteveen** (RAG tutorials)
  https://www.youtube.com/@samwitteveenai
  Practical RAG implementations with code walkthroughs.

- **All About AI**
  https://www.youtube.com/@AllAboutAI
  LLM news, RAG techniques, tool reviews.

---

## 8. Research & Academic Papers

Peer-reviewed research on RAG, hallucinations, and AI in libraries.

### Foundational RAG Papers

- **🎓 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**
  Lewis, P., et al. (2020). *NeurIPS 2020*.
  https://arxiv.org/abs/2005.11401
  Original RAG paper from Meta AI.

- **🎓 "A Survey on Hallucination in Large Language Models"**
  Huang, L., et al. (2025). *ACM Transactions on Information Systems*, 43(2).
  https://doi.org/10.1145/3703155
  Comprehensive review of hallucination research.

### Hallucinations & Limitations

- **🎓 "Hallucination is Inevitable: An Innate Limitation of Large Language Models"**
  Xu, Z., Jain, S., & Kankanhalli, M. (2024). *arXiv:2401.11817*.
  https://arxiv.org/abs/2401.11817
  Mathematical proof using learning theory. Cited in presentation.

- **🎓 "LLMs Will Always Hallucinate, and We Need to Live With This"**
  Banerjee, S., Agarwal, A., & Singla, S. (2024). *Springer LNNS*, vol 1554.
  https://doi.org/10.1007/978-3-031-99965-9_39
  Structural hallucination as intrinsic characteristic.

- **🎓 "Detecting Hallucinations Using Semantic Entropy"**
  Farquhar, S., et al. (2024). *Nature*, 630, 625-630.
  https://doi.org/10.1038/s41586-024-07421-0
  Detection methods published in Nature.

### Medical/Clinical Studies (High Hallucination Rates)

- **🎓 "Hallucination Rates and Reference Accuracy of ChatGPT and Bard"**
  Chelli, M., et al. (2024). *Journal of Medical Internet Research*, 26, e53164.
  https://doi.org/10.2196/53164
  28.6% (GPT-4), 39.6% (GPT-3.5), 91.4% (Bard) hallucination rates.

- **🎓 "Multi-model Assurance Analysis of LLM Vulnerabilities"**
  Omar, M., et al. (2025). *Communications Medicine*, 5, 330.
  https://doi.org/10.1038/s43856-025-01021-3
  Adversarial hallucination attacks in clinical decision support; reports 50-82% hallucination rates across models with mitigation strategies.

### Legal Domain Studies

- **🎓 "Legal RAG Hallucinations"** (Stanford RegLab)
  https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf
  Hallucination patterns in legal information retrieval.

### Technical Analysis & Industry Reports

- **"Why RAG Won't Solve Generative AI's Hallucination Problem"**
  Wiggers, K. (2024). *TechCrunch*.
  https://techcrunch.com/2024/05/04/why-rag-wont-solve-generative-ais-hallucination-problem/

- **"LLM Hallucinations in 2025"** (Lakera AI)
  https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models
  Current state of hallucination research and mitigation.

- **"AI Hallucinations Statistics"** (AllAboutAI)
  https://www.allaboutai.com/resources/ai-statistics/ai-hallucinations/
  Comprehensive hallucination rate data across models.

---

## 9. Evaluation Frameworks & Best Practices

Standards and checklists for assessing RAG systems.

### Professional Standards

- **📚⚡ ACRL AI Competencies for Academic Library Workers** (October 2025)
  https://www.ala.org/acrl/standards/ai
  Framework for evaluating AI tools, including RAG. Section 3.2 essential.

- **ACRL Framework for Information Literacy for Higher Education**
  https://www.ala.org/acrl/standards/ilframework
  Foundation for critical AI evaluation.

### Technical Evaluation

- **RAG Evaluation Metrics** (LlamaIndex)
  https://docs.llamaindex.ai/en/stable/examples/evaluation/
  Context relevance, answer relevance, faithfulness metrics.

- **🎓 RAGAS: Automated RAG Evaluation** (EACL 2024)
  **Paper:** https://aclanthology.org/2024.eacl-demo.16/
  **GitHub:** https://github.com/explodinggradients/ragas
  Essential for systematic RAG quality evaluation without expensive human annotation. Metrics: context precision, context recall, faithfulness, answer relevancy.

- **🆓 TruLens RAG Triad Evaluation** (TruEra)
  **Docs:** https://www.trulens.org/getting_started/core_concepts/rag_triad/
  **GitHub:** https://github.com/truera/trulens
  Helps identify hallucinations and ensure factual accuracy. Three metrics: context relevance, groundedness, answer relevance. Free course available from DeepLearning.AI.

- **🎓 RAGBench: Explainable RAG Benchmark** (July 2024)
  **Paper:** https://arxiv.org/abs/2407.11005
  First comprehensive large-scale RAG benchmark with 100K examples across 5 domains. TRACe evaluation framework with explainable metrics.

- **🎓 MIRAGE: Medical Information RAG Evaluation** (ACL 2024)
  **Website:** https://teddy-xionggz.github.io/benchmark-medical-rag/
  **GitHub:** https://github.com/Teddy-XiongGZ/MIRAGE
  Critical for health sciences libraries implementing RAG. Contains 7,663 medical questions from 5 datasets with MedRag toolkit.

### Ethical Considerations

- **"Knowledge Trade with Haves and Have-Nots"**
  Cox, C., & Tzoc, E. (2023). *College & Research Libraries News*.
  https://crln.acrl.org/index.php/crlnews/article/view/25868
  Equity concerns with tiered AI access. Cited in presentation.

---

## 10. RAG Security & Privacy

Critical considerations for patron-facing systems.

### Prompt Injection & Attacks

- **OWASP LLM Top 10: Prompt Injection**
  https://genai.owasp.org/llmrisk/llm01-prompt-injection/
  Industry-standard security guidelines for LLM applications.

- **"Indirect Prompt Injection: Generative AI's Greatest Security Flaw"** (Turing Institute)
  https://cetas.turing.ac.uk/publications/indirect-prompt-injection-generative-ais-greatest-security-flaw
  Explains how malicious documents can manipulate RAG outputs.

- **Microsoft Adaptive Prompt Injection Challenge**
  https://msrc.microsoft.com/blog/2024/12/announcing-the-adaptive-prompt-injection-challenge-llmail-inject/
  Demonstrates prompt injection worms in RAG systems (LLMail-Inject).

- **🎓 "RAG and Roll: Attacks on Retrieval-Augmented Generation"**
  https://arxiv.org/abs/2408.05025
  Research on RAG-specific attack vectors.

- **🎓 "Backdoor Attacks on RAG Systems"**
  https://arxiv.org/abs/2410.14479
  Document poisoning and adversarial attacks on RAG.

### Privacy Considerations

- **🎓 "Privacy in RAG Systems"** (ACL 2024 Findings)
  https://aclanthology.org/2024.findings-acl.267/
  RAG can leak private retrieval data but also mitigate LLM training data leakage.

### Key Concerns for Libraries

- **Data hygiene:** Malicious documents in knowledge base can poison outputs
- **Input validation:** User queries need sanitization
- **Content filtering:** Protect against adversarial prompts
- **Patron privacy:** Query logs reveal research interests
- **Access control:** Permission-aware retrieval essential

---

## 11. Community & Discussion

Where to ask questions and follow developments.

### Forums & Communities

- **r/LocalLLaMA** (Reddit)
  https://www.reddit.com/r/LocalLLaMA/
  Running models locally, privacy-focused RAG.

- **LangChain Discord**
  https://discord.com/invite/langchain
  Active community for RAG questions.

- **Hugging Face Forums**
  https://discuss.huggingface.co/
  Model discussions, embeddings, technical Q&A.

### Library-Specific

- **Code4Lib**
  https://code4lib.org/
  Library technology community. Mailing list and annual conference.

- **LITA (Library & Information Technology Association)**
  https://www.ala.org/lita/
  ALA division focused on library technology.

---

## 11. Cost Estimation & Budgeting

Realistic pricing for planning.

### API Costs (as of November 2025)

**OpenAI:**
- GPT-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- text-embedding-3-small: $0.02 per 1M tokens
- Typical RAG query: $0.001-0.005 depending on document size

**Anthropic:**
- Claude 3.5 Haiku: $0.25 per 1M input tokens, $1.25 per 1M output tokens
- Claude 3.5 Sonnet: $3 per 1M input tokens, $15 per 1M output tokens

**Pinecone:**
- Free tier: 100k vectors
- Starter ($70/month): 5M vectors, serverless
- Enterprise: Custom pricing

### Budget Planning

**Small pilot (100-1000 queries/month):**
- Free tier tools: $0
- Low-volume API: $5-20/month
- Vector DB: Free tier sufficient

**Medium deployment (10k queries/month):**
- API costs: $50-200/month
- Vector DB: $50-100/month
- Total: $100-300/month

**Production scale (100k+ queries/month):**
- API costs: $500-2000/month
- Vector DB: $200-1000/month
- Infrastructure: $500+/month
- Total: $1200-3500/month

**Hidden costs to factor:**
- Staff time (development, maintenance)
- Ethics review
- Training/documentation
- Monitoring and quality assurance

---

## 12. Take-Home Resources from Presentation

Materials specific to this presentation.

### Presentation Materials

- **GitHub Repository**
  https://github.com/radio-shaq/Lyrasis-slides-11-2025
  Slides, speaker notes, demo notebook, FAQ template

- **Working Colab Demo**
  `RAG_Demo_LYRASIS.ipynb` in repository
  Upload to https://colab.research.google.com/

- **Sample FAQ CSV Template**
  `sample_library_faq_data.csv` in repository
  Customize with your library's FAQs

### How to Use the Demo

1. Download `RAG_Demo_LYRASIS.ipynb` from GitHub repo
2. Upload to Google Colab
3. Run cells 1-5 (no API key needed for Part 1)
4. Optional: Add your OpenAI API key to try Part 2 (Playground)
5. Customize with your own FAQ CSV

### Evaluation Questions Checklist

Key questions from presentation for evaluating RAG vendors:

**Is it actually RAG?**
- Does it retrieve + cite documents?
- Can you see which chunks were retrieved?
- Does it cite sources before generating?

**Data & Privacy:**
- Where is data stored? (Cloud region?)
- Who has access?
- Used for training other models?
- GDPR/FERPA/accessibility compliance?

**Accuracy & Bias:**
- What are hallucination rates? (Demand numbers)
- How do they measure quality?
- Can you audit wrong answers?
- What happens when it's wrong?

**Cost & Lock-In:**
- Pricing model? (Per query? Per user? Per document?)
- What if usage 10x?
- Can you export data + embeddings?
- Contract terms?

---

## 13. What to Try This Week

Actionable next steps organized by time commitment.

### 30 Minutes

1. **Upload your FAQ to NotebookLM or ChatGPT**
   Watch how it retrieves and cites (or doesn't)

2. **Read ACRL AI Competencies Section 3.2**
   https://www.ala.org/acrl/standards/ai

3. **Browse Simon Willison's blog**
   https://simonwillison.net/

### 2-3 Hours

1. **Run the presentation Colab demo**
   Download from GitHub, customize with your FAQ CSV

2. **Try KingbotGPT** (SJSU chatbot)
   https://libapps.sjsu.edu/kingbot/
   Test with questions about your own library

3. **Read a hallucination research paper**
   Start with Xu et al. (2024) - it's accessible: https://arxiv.org/abs/2401.11817

### One Week

1. **Build a proof-of-concept**
   Use LlamaIndex tutorial + your FAQ data + free LLM

2. **Compare three no-code tools**
   NotebookLM vs ChatGPT vs Perplexity with same document set

3. **Draft an evaluation framework**
   Use ACRL competencies + questions checklist from presentation

---

## 14. Key Takeaways from Presentation

Quick reference to core concepts.

### What RAG Does

- **Without RAG:** LLM uses only training data (closed-book exam)
- **With RAG:** LLM retrieves your documents first (open-book exam)
- **Result:** Significantly reduces hallucinations (varies by model and task - see Chelli et al., Farquhar et al.)
- **Limitation:** Cannot eliminate hallucinations entirely (mathematically proven - see Xu et al., Banerjee et al.)

### Four Components Every RAG System Needs

1. **Document store** (your content)
2. **Embedding model** (converts text to vectors)
3. **Vector database** (semantic card catalog)
4. **LLM** (generates answer using retrieved docs)

Everything else is optional enhancement.

### Your Librarian Skills Applied to RAG

- **Source evaluation** → Which docs go in RAG system?
- **Cataloging/metadata** → How to structure knowledge?
- **Reference interviews** → Understanding user needs
- **Information literacy** → Teaching critical AI use
- **Workarounds** → Handling imperfect tools

### Hard Questions to Ask Before Deployment

- **Privacy:** Student queries reveal research interests, struggles
- **Equity:** Tiered access creates information haves/have-nots
- **Energy:** Training costs $1.5B-3B per model (The Information, 2025)
- **Censorship:** Who decides what goes in knowledge base?
- **Hallucinations:** Can reduce but never eliminate (peer-reviewed proof)

---

## 15. Stay in Touch & Keep Learning

### Contact

**David Meincke, MSLS**
Johnson Wales University
dmeincke@jwu.edu | davidmeincke@protonmail.com

Happy to discuss:
- Use case brainstorming
- Vendor demo review
- Technical troubleshooting
- Connecting with other library experimenters

### Update This Document

This resource list will evolve. Check the GitHub repository for updates:
https://github.com/radio-shaq/Lyrasis-slides-11-2025

### Contribute

Found a great resource not listed here? Open an issue or pull request on GitHub.

---

## Quick Reference: Decision Tree

**"Should I use RAG for this project?"**

```
START: What's your use case?

├─ Answering FAQs from your library's documents?
│  └─ YES → RAG is excellent for this
│     Recommended: Start with NotebookLM or Colab demo
│
├─ Discovery layer enhancement?
│  └─ MAYBE → Consider vendor solutions (Primo, Summon, EBSCO)
│     Recommended: Pilot before committing
│
├─ Citation/research help?
│  └─ YES → RAG works well with proper source verification
│     Recommended: LlamaIndex + OpenAI + human oversight
│
├─ General knowledge questions?
│  └─ NO → Standard LLM probably better
│     Recommended: ChatGPT, Claude without RAG
│
└─ Sensitive/confidential information?
   └─ MAYBE → Only with local models + privacy audit
      Recommended: Ollama + local embeddings + ethics review
```

---

**Last updated:** November 5, 2025
**License:** MIT (free to use, adapt, share with attribution)

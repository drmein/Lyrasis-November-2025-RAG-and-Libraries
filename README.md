# Supplemental Resources: RAG for Libraries

**Companion to LYRASIS Presentation: "Giving AI Access to Library Knowledge"**
**David Meincke, MSLS | Johnson & Wales University | November 2025**

---

## 📖 What Is This?

**TLDR:** A curated guide to help librarians explore, evaluate, and implement RAG (Retrieval-Augmented Generation) systems. Organized for library professionals of all technical levels, from "never coded" to "systems librarian."

**What's RAG?** Think of it as "open-book exam for AI." Instead of relying on training data alone, RAG systems retrieve YOUR documents (policies, FAQs, catalogs) and cite them when answering questions. This makes AI outputs more grounded, verifiable, and tied to your actual library knowledge.

**Here's the thing:** You already have RAG expertise. Source evaluation, cataloging, reference work—these skills map directly to RAG evaluation. This guide helps you apply what you already know to a new tool.

**Key:**
- 🆓 Free/Open Source
- 💰 Paid/Commercial (with free tiers noted)
- 📚 Library-Specific
- 🎓 Academic/Peer-Reviewed
- ⚡ Quick Start (try today)

---

## ⭐ Try This Afternoon (No Code Required)

Start here if you want to *experience* RAG right now, no technical skills needed.

### 🟢 NotebookLM (Google's Free RAG Tool)

**URL:** https://notebooklm.google.com/
**Why start here:** Upload your library's FAQ PDF, ask it questions, watch it cite sources. This is RAG in action. Free, never used to train models.

**What it does:**
- Upload up to 50 sources per notebook (PDFs, Google Docs, websites, audio)
- Ask questions → get answers with citations
- Generate study guides, timelines, FAQs
- Create podcast-style audio summaries
- Share notebooks with colleagues

**Privacy note:** Google states your data isn't used for model training. Still, don't upload confidential patron data.

**Real library use case:** Upload your library policy manual. Ask "What's our laptop checkout policy?" Watch it cite the exact section. Now you understand RAG.

**Tutorials:**
- Official Google Guide: https://blog.google/technology/ai/notebooklm-beginner-tips/
- DataCamp Tutorial: https://www.datacamp.com/tutorial/notebooklm

### 🟢 ChatGPT with PDF Upload

**URL:** https://chat.openai.com/
**Free tier:** 3 PDFs per day (as of 2025)
**Why try it:** Upload a document, ask questions, notice the citations. That's RAG.

**Limitation:** Smaller context than NotebookLM, no persistent knowledge base. But good for understanding the basic pattern: retrieve → cite → generate.

### 🟢 Perplexity AI

**URL:** https://www.perplexity.ai/
**Why it's useful:** Web-based RAG with live citations. Good for seeing how source cards work in a clean interface.

**Your advantage as a librarian:** You evaluate sources for a living. When you try these tools, you'll immediately spot when citations are weak, when retrieval is poor, when answers aren't grounded. That's your professional expertise at work.

---

## 📚 Learn from Peer Libraries

Real examples from libraries like yours. Use these as templates and inspiration.

### ⭐ KingbotGPT (San José State University)

**Live demo:** https://library.sjsu.edu/kingbot/
**Launch:** September 2024
**Tech Stack:** LlamaIndex, OpenAI GPT-4o-mini, ChromaDB, Streamlit
**Open Source:** Yes, code on GitHub

**Why it's important:** Complete technical case study from an academic library. Built by librarians to understand RAG deeply enough to evaluate vendor tools critically.

**Cost reality:** Minimal monthly API costs (likely $20-40/month for production use)

**Key insight from their paper:** Development was designed for "capacity building"—they built it so librarians could evaluate vendor AI tools from an informed position, not just accept marketing claims.

**Paper:** "Library-Led AI: Building a Library Chatbot as Service and Strategy" (ACRL 2025)
https://www.ala.org/sites/default/files/2025-03/Library-LedAI.pdf

**Lesson learned:** They switched from LangChain to LlamaIndex ("simpler code that is easier to maintain"). Hybrid retrieval: website crawl + custom Q&A knowledge base.

### T-Rex Chatbot (University of Calgary)

**Launch:** August 2021 (mature implementation)
**Tech:** Ivy.ai RAG + LLMs
**Performance:** Answers ~50% of questions at 4/5 rating or higher, freed 1.5 FTE for higher-level work
**Languages:** 100+ supported
**Accessibility:** WCAG 2.1 AA compliant

**Why it matters:** Longest-running library RAG chatbot with published metrics. Shows what "production-ready" looks like after years of refinement.

**Article:** "Implementing an AI Reference Chatbot at University of Calgary Library" (OCLC Research, Nov 2024)
https://hangingtogether.org/implementing-an-ai-reference-chatbot-at-the-university-of-calgary-library/

### Vendor Discovery Systems with RAG

**All launched 2024-2025, all opt-in:**

- **📚💰 Ex Libris Primo Research Assistant** (Sept 2024)
  RAG grounded in 5+ billion CDI records
  https://clarivate.com/news/clarivate-launches-generative-ai-powered-primo-research-assistant/

- **📚💰 ProQuest Summon Research Assistant** (March 2025)
  Based on Ex Libris RAG architecture
  https://librarytechnology.org/pr/31135

- **📚💰 EBSCO AI Insights** (March 2025)
  RAG grounded on full-text articles
  https://www.ebsco.com/news-center/press-releases/ebsco-information-services-launches-newest-artificial-intelligence-features/

**Critical note from Aaron Tay (systems librarian, widely followed):** Several vendor systems have guardrail problems, blocking legitimate searches like "Gaza war" or "Tulsa race riot." Ask vendors how they handle sensitive topics. This is where your professional judgment matters.

### LibGuides from Peer Libraries

Real examples you can adapt:

- **⭐ Ohio State Lima - Artificial Intelligence LibGuide**
  https://osu.libguides.com/lima_ai
  Explains RAG in plain language, covers practical tools, includes citation guidance

- **Florida International University - AI + ACRL Frameworks**
  https://library.fiu.edu/AI-ACRL/intro
  Maps AI tools to ACRL Framework for Information Literacy—perfect for instruction librarians

- **University of Alaska Southeast - Artificial Intelligence**
  https://uas.alaska.libguides.com/ai
  Clear definitions using industry standards (NIST, IBM, NVIDIA), good glossary

---

## 🧭 Essential Framework: Start Here

Before diving into tools or vendors, read this. It's your north star.

### ⭐ ACRL AI Competencies for Academic Library Workers

**URL:** https://www.ala.org/acrl/standards/ai
**Approved:** October 2025 (just released!)

**Why it matters:** **This is THE framework.** Four competency areas: 
1. Ethical Considerations
2. Knowledge & Understanding
3. Analysis & Evaluation
4. Use & Application

**Key insight:** ACRL positions librarians as *critical evaluators* of AI tools, not passive users. Uses language like "grounded and verifiable" for good RAG outputs.

**Your existing skills map directly:**
- Source evaluation → Which docs go in RAG system?
- Cataloging/metadata → How to structure knowledge?
- Reference interviews → Understanding user needs
- Information literacy instruction → Teaching critical AI use

**How to use it:** Reference Section 3.2 when evaluating vendor tools. Use it to justify pilot projects to administration. Cite it when explaining why librarians should lead AI evaluation on campus.

---

## ✅ Ethics & Privacy: Before You Deploy

Ask these questions BEFORE buying or building anything. Have conversations with your team, not just technical checklists.

### Privacy: What Are We Collecting?

**Remember:** Student queries to chatbots reveal research struggles, health questions, personal challenges. Treat chat logs like reference desk conversations—confidential and requiring protection.

**Questions to discuss:**
- What's our data retention policy for queries?
- Who has access to query logs?
- How do we handle queries that reveal sensitive research topics?
- What happens if we get a subpoena for chat logs?

**⭐ ALA Library Privacy Checklist**
https://www.ala.org/advocacy/privacy/checklists/overview
Three-tier priority system. Priority 1 = all libraries can do right now.

**AI-Specific Privacy Checklist (Just Solutions)**
https://www.justinc.com/blog/ai-privacy-and-security-checklist/
Key principle: "Think of it like a public forum. If you wouldn't post it on your website, don't type it into a chatbot."

### Equity: Who Benefits?

Premium tiers = better answers. Are you creating information haves and have-nots?

**Article:** "Knowledge Trade with Haves and Have-Nots" - Cox & Tzoc (2023)
https://crln.acrl.org/index.php/crlnews/article/view/25868
Real talk about tiered AI access in libraries.

### Questions to Discuss (No Easy Answers)

Before you launch a RAG system, have these conversations with your team:

- **Censorship:** Who decides what goes in the knowledge base? What gets excluded? How do we handle controversial topics?
- **Transparency:** Will you disclose when students are talking to AI vs. humans? Does it matter?
- **Accessibility:** Is your RAG system WCAG compliant? Screen reader compatible?
- **Hallucinations:** RAG reduces errors significantly but can't eliminate them (mathematically proven—see research section). How do you communicate this limitation honestly?
- **Environmental cost:** Training frontier models costs $1.5-3 billion per model (The Information, 2025). What's our responsibility here?

**Hard truth from research:** RAG can reduce hallucinations dramatically (studies show 15-80% improvement depending on implementation), but Xu et al. (2024) mathematically proved eliminating hallucinations entirely is impossible. Plan for errors, not perfection.

---

## 🤔 Questions to Ask RAG Vendors

Apply your source evaluation skills to sales pitches. Demand proof, not marketing.

### Is It Actually RAG?

Not every "AI-powered" tool uses RAG. Many are just search-only or summarization-only.

- [ ] Does it retrieve documents AND cite sources?
- [ ] Can you see which chunks were retrieved?
- [ ] Does it cite sources before generating answers?
- [ ] Can you test it with your own documents?

**Red flag:** If they say "AI-powered" but can't show you citations, it's probably not RAG.

### Privacy & Data

- [ ] Where is our library data stored? (Which country? Cloud region?)
- [ ] Who has access to patron queries?
- [ ] Is our data used to train your models?
- [ ] FERPA/GDPR/accessibility compliance documentation?
- [ ] What happens to chat logs? How long retained?
- [ ] Can we audit what data you have about our users?
- [ ] What's your security model? (See OWASP LLM Top 10)

### Accuracy & Bias

- [ ] What are your measured hallucination rates? (Demand actual numbers, not "very low")
- [ ] How do you evaluate quality? (Ask for their testing methodology)
- [ ] Can we audit wrong answers? (Access to logs, feedback mechanisms)
- [ ] What happens when it's wrong? (Error handling, user notifications)
- [ ] How do you handle bias in retrieval/generation?
- [ ] How do you handle controversial or sensitive topics?

### Cost & Lock-In

- [ ] Pricing model? (Per query? Per user? Per document? Per token?)
- [ ] What if our usage increases 10x? (Scalability costs)
- [ ] Can we export our data + embeddings? (Or are we locked in forever?)
- [ ] Contract terms? (Auto-renewal? Cancellation policy?)
- [ ] Hidden costs? (Training, customization, API overages, maintenance)

### Performance & Reliability

- [ ] What's your uptime SLA?
- [ ] Average response time? (2 seconds? 10 seconds? Users abandon slow tools)
- [ ] How often do models update? (Can we control this or does it just happen?)
- [ ] What happens during outages?

### Support & Training

- [ ] What training do you provide our staff?
- [ ] Ongoing support structure? (Email? Phone? Ticket system?)
- [ ] Documentation quality? (Can we see it before purchase?)
- [ ] Community/user group?

**Remember:** You evaluate sources professionally. Apply that same rigor here. If their answers are vague or defensive, that's a data point.

---

## 🛠️ Build Your Own? Tools by Experience Level

### 🟢 No Code (This Afternoon)

Already covered above—start with NotebookLM. You can have a working RAG system in 30 minutes with zero coding.

### 🟡 Low Code (Weekend Project)

**Google Colab + Pre-built Notebooks**
- No local setup needed (runs in browser)
- Free tier available
- Modify existing RAG notebooks slightly
- Good for learning and demos

**Streamlit Chatbot Builder**
- Python-based UI framework
- KingbotGPT uses this
- Free Community Cloud hosting
- Tutorial: https://streamlit.io/

**Presentation demo materials:**
- GitHub: https://github.com/radio-shaq/Lyrasis-slides-11-2025
- Includes working Colab notebook you can customize
- Sample FAQ template provided

### 🔴 Full Implementation (Systems Librarians)

**Python RAG Frameworks:**

**🆓 LlamaIndex** (Recommended for beginners)
- https://www.llamaindex.ai/
- Purpose-built for RAG
- Excellent documentation
- Used by SJSU KingbotGPT
- Tutorial: https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html

**🆓 LangChain** (More general-purpose)
- https://www.langchain.com/
- Broader LLM orchestration
- Larger ecosystem
- Tutorial: https://python.langchain.com/docs/tutorials/rag/

**Vector Databases (Your Semantic Card Catalog):**

**🆓 ChromaDB** (Start here for prototypes)
- https://www.trychroma.com/
- Embedded, no separate server needed
- Free forever
- Used in presentation demo + KingbotGPT

**🆓 PostgreSQL pgvector** ⭐ Major 2024 development
- Tutorial: https://www.enterprisedb.com/blog/rag-app-postgres-and-pgvector
- **Huge for libraries already using PostgreSQL** for ILS/discovery systems
- Reduces infrastructure complexity—one database for everything
- Free if you're already running Postgres

**🆓 Weaviate / Qdrant / FAISS**
- All solid open-source options
- Free tiers available for cloud versions
- Production-ready

**💰 Pinecone** (If you need managed/scaled)
- https://www.pinecone.io/
- Free tier: 100k vectors (plenty for pilots!)
- Starter: $70/month for 5M vectors
- Only upgrade if you outgrow free tier

**Large Language Models:**

**💰 OpenAI API** (GPT-4o-mini recommended)
- https://platform.openai.com/
- **Cost:** $0.15 per 1M input tokens, $0.60 per 1M output tokens
- **Reality check:** 1000 RAG queries/month = ~$0.60. Yes, really.

**💰 Anthropic API** (Claude 3.5 Haiku)
- https://www.anthropic.com/api
- Strong context windows (200k+ tokens), good for long documents
- Similar pricing to OpenAI

**🆓 Ollama** (Run models locally)
- https://ollama.ai/
- Run Llama, Mistral, Phi locally
- **Privacy-focused:** Everything stays on your server
- Free, but needs decent hardware (GPU helpful)

**🆓 Llama 3.2 & 3.3** (Meta AI, Sept-Dec 2024)
- https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- Enables on-premise/privacy-preserving RAG
- Lightweight models (1B, 3B) can run on standard library hardware
- Vision models (11B, 90B) for multimodal RAG

---

## 🆕 What's New in 2024-2025?

Developments worth knowing about.

### GraphRAG (Microsoft, October 2024) ⭐ Game-Changer

**What it is:** Uses LLMs to build knowledge graphs from your documents, then does RAG on the graph instead of just text chunks.

**Why it matters:** 70-80% improvement over standard RAG for complex questions. **Exceptional for special collections, archives, and complex document relationships.**

**Resources:**
- Main: https://www.microsoft.com/en-us/research/project/graphrag/
- GitHub: https://github.com/microsoft/graphrag
- Blog: https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/

**Use case:** Your archives have hundreds of interrelated documents. GraphRAG understands the relationships between them, not just keyword matches.

### Reranking (Essential for Production)

**What it is:** After initial retrieval, a reranker re-scores results by processing query + document together.

**Why it matters:** 13-25% accuracy improvement over embeddings alone. Makes the difference between "good enough for demo" and "good enough for production."

**Tools:**
- **💰 Cohere Rerank v3.5** (Dec 2024, state-of-the-art)
  https://cohere.com/rerank
  100+ languages, 4096 token context
  
- **🆓 Cross-Encoders** (Sentence Transformers)
  https://www.sbert.net/examples/applications/cross-encoder/README.html
  Open-source alternative

**Article:** "The Missing Component in Your RAG Chatbot" (Coalfire)
https://coalfire.com/the-coalfire-blog/one-component-you-desperately-need-in-your-rag-chatbot-toolchain

### AutoRAG (October 2024)

**What it is:** AutoML-style automation for RAG pipelines. Tests different configurations automatically to find what works best.

**Why it matters:** Simplifies RAG development for libraries without ML expertise.

**Resources:**
- Paper: https://arxiv.org/abs/2410.20878
- GitHub: https://github.com/Marker-Inc-Korea/AutoRAG

### NotebookLM Audio Overviews (October 2024)

Generate podcast-style discussions from your documents. Two AI hosts talk through your content. Wild feature for consuming content while multitasking.

**Use case:** Turn your library's annual report into a 10-minute podcast for busy staff.

---

## 📊 Evaluation & Testing

How to know if your RAG system actually works.

### Professional Standards

**📚⚡ ACRL AI Competencies** (October 2025)
https://www.ala.org/acrl/standards/ai
Use Section 3.2 as evaluation rubric

**ACRL Framework for Information Literacy**
https://www.ala.org/acrl/standards/ilframework
Foundation for critical AI evaluation

### Technical Evaluation Frameworks

**🆓 RAGAS: Automated RAG Evaluation** (EACL 2024)
- Paper: https://aclanthology.org/2024.eacl-demo.16/
- GitHub: https://github.com/explodinggradients/ragas
- **Why it matters:** Systematic quality evaluation without expensive human annotation
- Metrics: context precision, context recall, faithfulness, answer relevancy

**🆓 TruLens RAG Triad Evaluation** (TruEra)
- Docs: https://www.trulens.org/getting_started/core_concepts/rag_triad/
- GitHub: https://github.com/truera/trulens
- Helps identify hallucinations, ensure factual accuracy
- Three metrics: context relevance, groundedness, answer relevance
- Free course available from DeepLearning.AI

**🎓 RAGBench** (July 2024)
- Paper: https://arxiv.org/abs/2407.11005
- First comprehensive large-scale RAG benchmark
- 100K examples across 5 domains

**🎓 MIRAGE: Medical Information RAG Evaluation** (ACL 2024)
- https://teddy-xionggz.github.io/benchmark-medical-rag/
- **Critical for health sciences libraries implementing RAG**
- 7,663 medical questions from 5 datasets

### What to Test

From LlamaIndex/TruLens frameworks:

**Context Relevance:** Are the retrieved documents actually relevant to the query?
**Answer Relevance:** Does the answer actually address what was asked?
**Faithfulness/Groundedness:** Is the answer supported by the retrieved documents?

**Your library test set:**
- Create 50-100 Q&A pairs from your FAQ
- Run them through your RAG system
- Manually review answers for accuracy
- Track hallucination rate
- Compare to vendor claims

---

## 🔒 Security & Privacy Concerns

Critical for patron-facing systems.

### Prompt Injection Attacks

**The problem:** Malicious documents in your knowledge base can manipulate RAG outputs. User queries can contain adversarial prompts.

**Resources:**
- **OWASP LLM Top 10: Prompt Injection**
  https://genai.owasp.org/llmrisk/llm01-prompt-injection/
  Industry-standard security guidelines

- **"Indirect Prompt Injection"** (Turing Institute)
  https://cetas.turing.ac.uk/publications/indirect-prompt-injection-generative-ais-greatest-security-flaw
  How malicious documents manipulate RAG outputs

- **Microsoft Adaptive Prompt Injection Challenge**
  https://msrc.microsoft.com/blog/2024/12/announcing-the-adaptive-prompt-injection-challenge-llmail-inject/
  Demonstrates prompt injection worms in RAG systems

- **🎓 "RAG and Roll: Attacks on Retrieval-Augmented Generation"**
  https://arxiv.org/abs/2408.05025
  Research on RAG-specific attack vectors

### Key Library Concerns

**Data hygiene:** Malicious documents in knowledge base can poison outputs
**Input validation:** User queries need sanitization
**Content filtering:** Protect against adversarial prompts
**Patron privacy:** Query logs reveal research interests
**Access control:** Permission-aware retrieval essential

**Privacy research:**
- **🎓 "Privacy in RAG Systems"** (ACL 2024 Findings)
  https://aclanthology.org/2024.findings-acl.267/
  RAG can leak private retrieval data but also mitigate LLM training data leakage

---

## 📖 Research Papers: The Science Behind RAG

### Foundational RAG

**🎓 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**
Lewis, P., et al. (2020). *NeurIPS 2020*.
https://arxiv.org/abs/2005.11401
Original RAG paper from Meta AI. Where it all started.

### Hallucinations & Limitations (Why RAG Can't Be Perfect)

**🎓 "Hallucination is Inevitable: An Innate Limitation of Large Language Models"**
Xu, Z., Jain, S., & Kankanhalli, M. (2024). *arXiv:2401.11817*.
https://arxiv.org/abs/2401.11817
**Mathematical proof** using learning theory. Hallucinations cannot be eliminated for any computable LLM, regardless of architecture or training. Cited in presentation.

**🎓 "LLMs Will Always Hallucinate, and We Need to Live With This"**
Banerjee, S., Agarwal, A., & Singla, S. (2024). *Springer LNNS*, vol 1554.
https://doi.org/10.1007/978-3-031-99965-9_39
Structural hallucination as intrinsic characteristic using Gödel's Incompleteness Theorem.

**🎓 "Detecting Hallucinations Using Semantic Entropy"**
Farquhar, S., et al. (2024). *Nature*, 630, 625-630.
https://doi.org/10.1038/s41586-024-07421-0
Detection methods published in *Nature*. Shows how to identify when LLMs are uncertain.

### Measured Hallucination Rates

**🎓 "Hallucination Rates and Reference Accuracy of ChatGPT and Bard"**
Chelli, M., et al. (2024). *Journal of Medical Internet Research*, 26, e53164.
https://doi.org/10.2196/53164
**Without RAG:** 28.6% (GPT-4), 39.6% (GPT-3.5), 91.4% (Bard) hallucination rates
Shows the importance of RAG for reducing errors.

**🎓 "Multi-model Assurance Analysis of LLM Vulnerabilities"**
Omar, M., et al. (2025). *Communications Medicine*, 5, 330.
https://doi.org/10.1038/s43856-025-01021-3
50-82% hallucination rates across models in clinical settings; mitigation strategies.

### Library-Specific Research

**🎓 "Prospects of RAG for Academic Library Search and Retrieval"**
Bevara, R.V.K., et al. (2025). *Information Technology and Libraries*, 44(2).
https://doi.org/10.5860/ital.v44i2.17361
Academic analysis of RAG applications in library contexts.

### Survey Papers (Great Overviews)

**🎓 "A Survey on Hallucination in Large Language Models"**
Huang, L., et al. (2025). *ACM Transactions on Information Systems*, 43(2).
https://doi.org/10.1145/3703155
Comprehensive review of hallucination research, mitigation strategies.

---

## 💰 Real Talk About Costs

Let's be honest about what this actually costs, not vendor fantasy numbers.

### The SJSU Reality Check

KingbotGPT at San José State runs in production serving students 24/7. Their costs are **minimal**—likely $20-40/month total for APIs. They're not spending hundreds or thousands per month.

### Realistic Budget Scenarios

**Proof-of-Concept / Small Pilot (500-1000 queries/month)**
- OpenAI GPT-4o-mini: $0.50-3/month
- ChromaDB: Free (embedded)
- Sentence Transformers embeddings: Free (open-source)
- Streamlit Community Cloud: Free tier
- **Total: $0-5/month**

Seriously. You can run a working RAG system for the cost of a latte.

**Small Production (5000-10k queries/month)**
- API costs: $5-30/month
- Vector database: Free tier (ChromaDB/Weaviate)
- Hosting: Free tier (Streamlit/Hugging Face Spaces)
- **Total: $10-50/month**

**Medium Scale (50k queries/month)**
- API costs: $50-150/month
- Vector DB: $0-70/month (may still be free tier!)
- **Total: $50-200/month**

### Cost Breakdown Example

**1000 student queries per month:**
- Average query: ~500 input tokens (retrieved context)
- Average response: ~150 output tokens
- GPT-4o-mini pricing: $0.15 per 1M input, $0.60 per 1M output

**Math:**
- Input: 500K tokens × $0.15/1M = $0.075
- Output: 150K tokens × $0.60/1M = $0.090
- **Total: $0.17/month**

Add embeddings and you're still under $1/month for 1000 queries.

### Free Tier Reality

**You can pilot RAG entirely free:**
- NotebookLM: Free
- ChromaDB: Free
- Sentence Transformers: Free
- OpenAI free tier: 3 requests/day (for testing)
- Ollama + local models: Free (if you have hardware)
- Streamlit Community Cloud: Free
- Google Colab: Free

**When you need to pay:**
- Scaling beyond ~10k queries/month
- Need faster response times
- Want commercial support
- Need enterprise features (SSO, audit logs, SLAs)

### Hidden Costs to Actually Consider

The expensive part isn't the tools, it's:
- **Staff time** (development, maintenance, training)
- **Ethics review** (getting institutional buy-in)
- **Quality assurance** (ongoing testing, monitoring)
- **Documentation** (training materials, user guides)

Budget staff time, not API costs.

---

## 🌐 Who to Follow & Where to Learn

### Blogs (Highly Recommended)

**⚡ Aaron Tay's Musings about Librarianship**
- Substack: https://aarontay.substack.com/
- Blog: http://musingsaboutlibrarianship.blogspot.com/
- **Who:** Head of Data Services, Singapore Management University
- **Why follow:** THE library voice on AI/RAG. Weekly vendor reviews, critical analysis, real implementations. On Clarivate AI governance board.

**⚡ Simon Willison's Weblog**
- https://simonwillison.net/
- **Who:** Co-creator of Django, AI researcher
- **Why follow:** Daily RAG experiments, practical tools, critical thinking. Not library-specific but essential for understanding what's actually possible.

**LlamaIndex Blog**
https://www.llamaindex.ai/blog
RAG techniques, case studies, tutorials.

**Pinecone Learning Center**
https://www.pinecone.io/learn/
Vector database concepts, RAG patterns, excellent explanations.

### Tutorials & Courses (Free)

**🆓 "Building RAG Applications" (LangChain)**
https://python.langchain.com/docs/tutorials/rag/
Step-by-step tutorial with code

**🆓 "RAG Tutorial" (LlamaIndex)**
https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html
Quickstart with clear examples

**💰 "LangChain for LLM Application Development"** (DeepLearning.AI)
https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/
Free short course by Andrew Ng

### Community Forums

**r/LocalLLaMA** (Reddit)
https://www.reddit.com/r/LocalLLaMA/
Running models locally, privacy-focused RAG

**LangChain Discord**
https://discord.com/invite/langchain
Active community for RAG questions

**Code4Lib**
https://code4lib.org/
Library technology community, mailing list, annual conference

**LITA (Library & Information Technology Association)**
https://www.ala.org/lita/
ALA division focused on library technology

---

## 📋 What to Try This Week

Actionable next steps by time commitment.

### 30 Minutes

**1. Upload your FAQ to NotebookLM**
- Go to https://notebooklm.google.com/
- Upload your library FAQ (PDF, Google Doc, or paste text)
- Ask 5 questions you know the answers to
- Watch how it retrieves and cites (or doesn't)
- **You now understand RAG UX patterns**

**2. Read ACRL AI Competencies Section 3.2**
https://www.ala.org/acrl/standards/ai
The evaluation framework you need

**3. Try KingbotGPT**
https://library.sjsu.edu/kingbot/
Ask it questions about your own library (it won't know, but watch how it responds)

### 2-3 Hours

**1. Run the presentation Colab demo**
- Download from GitHub: https://github.com/radio-shaq/Lyrasis-slides-11-2025
- Upload `RAG_Demo_LYRASIS.ipynb` to Google Colab
- Customize with your FAQ CSV
- **You now understand RAG architecture**

**2. Compare three tools with the same document**
- Upload your circulation policy to NotebookLM, ChatGPT, and Perplexity
- Ask the same 10 questions to each
- Compare citation quality
- **You now understand RAG quality differences**

**3. Read a hallucination research paper**
Start with Xu et al. (2024) - it's surprisingly accessible:
https://arxiv.org/abs/2401.11817

### One Week

**1. Build a proof-of-concept**
- Use LlamaIndex tutorial + your FAQ data + OpenAI free tier
- Get something working end-to-end
- **You now understand RAG implementation**

**2. Draft an evaluation framework**
- Use ACRL competencies
- Add vendor questions from this guide
- Customize for your library's needs
- **You now have a tool for vendor demos**

**3. Have the ethics conversation**
- Share the ethics section with your team
- Discuss: What are our lines? What makes us uncomfortable?
- Document your library's values
- **You now have a foundation for AI policy**

---

## 🎯 Key Takeaways

Quick reference to core concepts from the presentation.

### What RAG Does

**Without RAG:** LLM uses only training data (closed-book exam)
**With RAG:** LLM retrieves your documents first (open-book exam)
**Result:** Dramatically reduces hallucinations (studies show 70-90%+ improvement)
**Limitation:** Cannot eliminate hallucinations entirely (mathematically proven)

### Four Components Every RAG System Has

1. **Document store** (your content—FAQs, policies, guides)
2. **Embedding model** (converts text to vectors for semantic search)
3. **Vector database** (semantic card catalog that finds similar content)
4. **LLM** (generates answer using retrieved documents)

Everything else is enhancement or optimization.

### Your Librarian Skills = RAG Expertise

- **Source evaluation** → Curating RAG knowledge base
- **Cataloging/metadata** → Structuring documents for retrieval
- **Reference interviews** → Understanding user information needs
- **Information literacy** → Teaching critical AI evaluation
- **Workarounds** → Making imperfect systems work

You're not learning something completely new. You're applying expertise you already have.

### Hard Questions With No Easy Answers

- **Privacy:** Queries reveal research struggles, health questions, identity explorations
- **Equity:** Premium tiers create information haves/have-nots
- **Energy:** Training frontier models costs billions, significant environmental impact
- **Censorship:** Who decides what goes in the knowledge base? What gets excluded?
- **Hallucinations:** Can reduce dramatically but never eliminate completely
- **Displacement:** Does this replace library jobs or augment them?

Have these conversations before you deploy, not after.

### What's Actually Possible vs. Marketing Hype

**Possible:**
- FAQ chatbots with good citation (proven, working today)
- Document search with semantic understanding (works well)
- Research assistance with source grounding (improving rapidly)
- Reducing hallucinations 70-90% compared to no RAG (peer-reviewed)

**Not possible (yet or ever):**
- 100% accuracy (mathematically impossible)
- Perfect understanding of user intent (humans struggle with this too)
- Replacing librarian judgment (tools need expert curation)
- Eliminating bias (LLMs reflect training data biases)

Be honest about what RAG can and can't do. Under-promise and over-deliver.

---

## 🔍 Decision Tree: Should You Use RAG?

```
START: What's your use case?

├─ Answering FAQs from library documents?
│  └─ YES → RAG is excellent for this
│     Start with: NotebookLM or presentation Colab demo
│     Budget: $0-5/month pilot, $10-50/month production
│
├─ Discovery layer enhancement?
│  └─ MAYBE → Consider vendor solutions (Primo, Summon, EBSCO)
│     Action: Pilot before committing, use ACRL framework for evaluation
│     Budget: Already paying for discovery layer
│
├─ Citation/research help?
│  └─ YES → RAG works well with proper oversight
│     Start with: LlamaIndex + OpenAI + human verification
│     Budget: $20-100/month depending on usage
│
├─ General knowledge questions?
│  └─ NO → Standard LLM probably better
│     Why: No need for retrieval if not using specific documents
│     Alternative: ChatGPT, Claude without RAG
│
└─ Sensitive/confidential information?
   └─ MAYBE → Only with local models + privacy audit
      Requirements: Ollama + local embeddings + ethics review
      Budget: Hardware costs instead of API costs
```

---

## 📧 Stay in Touch & Keep Learning

### Contact

**David Meincke, MSLS**
Johnson & Wales University
dmeincke@jwu.edu | davidmeincke@protonmail.com

Happy to discuss:
- Use case brainstorming
- Vendor demo review
- Technical troubleshooting
- Connecting with other library experimenters

### Presentation Materials

**GitHub Repository:**
https://github.com/radio-shaq/Lyrasis-slides-11-2025

Contains:
- Slides + speaker notes
- Working Colab demo notebook
- Sample FAQ template CSV
- This resource guide

### Keep This Document Updated

RAG is evolving rapidly. Check GitHub for updates, or fork and maintain your own version.

**Contribute:** Found a great resource not listed? Open an issue or pull request.

---

## 🙏 Final Thoughts

Remember: You already have the skills to evaluate RAG systems critically. Your professional judgment as a librarian—your ability to assess sources, understand user needs, navigate ambiguity, and make imperfect systems work—is exactly what's needed to deploy AI thoughtfully in libraries.

RAG isn't magic. It's a tool, like any other library technology. It has strengths (grounded answers, source citation) and limitations (hallucinations, bias, cost). Your job is the same as always: evaluate critically, implement thoughtfully, and keep the needs of your community at the center.

Start small. Test thoroughly. Be honest about limitations. Keep learning.

You've got this.

---

**Last updated:** November 5, 2025
**License:** MIT (free to use, adapt, share with attribution)
**Version:** 3.0 (synthesis)

---

**Quick Links (Most Important Resources):**
- ⚡ Try today: https://notebooklm.google.com/
- 📚 Framework: https://www.ala.org/acrl/standards/ai
- 💻 Demo code: https://github.com/radio-shaq/Lyrasis-slides-11-2025
- 📖 Follow: https://aarontay.substack.com/
- 🔬 Research: https://arxiv.org/abs/2401.11817

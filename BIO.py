import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd

df_chunks = pd.read_csv("Bio_curriculum_chunks1000_over20.csv")
index = faiss.read_index("Bio_curriculum_faiss_index_1000_over20.bin")

Emb_model = SentenceTransformer("intfloat/multilingual-e5-base")
def score_query(query, threshold=0.79, k=5):
    """
    Scores a query against the FAISS index and determines if it's in the curriculum.
    
    Parameters:
    - query (str): The input query to be scored.
    - threshold (float): The score threshold to determine if in curriculum.
    - k (int): The number of top chunks to retrieve from the index.
    
    Returns:
    - dict: Dictionary containing score, in_curriculum boolean, and top chunks
    """
    # Encode the query into the same embedding space
    query_emb = Emb_model.encode(["query: " + query], convert_to_numpy=True)
    
    # Normalize the query embedding (L2 normalization)
    faiss.normalize_L2(query_emb)
    
    # Search the FAISS index
    D, I = index.search(query_emb, k)
    
    # Get the top score (similarity score)
    score = float(D[0][0])
    in_curriculum = score >= threshold
    
    # Get top chunks
    top_chunks = []
    for i in range(min(k, len(I[0]))):
        if I[0][i] < len(df_chunks):
            chunk = {
                'text': df_chunks.iloc[I[0][i]]['text'],
                'score': float(D[0][i]),
                'metadata': {
                    'lesson': df_chunks.iloc[I[0][i]].get('lesson', 'Unknown'),
                    'chapter': df_chunks.iloc[I[0][i]].get('chapter', 'Unknown')
                }
            }
            top_chunks.append(chunk)
    
    return {
        'score': score,
        'in_curriculum': in_curriculum,
        'top_chunks': top_chunks,
        'status': "in scope✅" if in_curriculum else "out of scope🚫"  
    }
def retrieve_context(query: str, k: int = 5):
    """
    Retrieve relevant context chunks for a given query.
    
    Args:
        query: User's question
        k: Number of chunks to retrieve
        
    Returns:
        List of formatted context chunks and similarity score
    """
    # Assess query against curriculum
    query_result = score_query(query, threshold=0.79, k=k)
    
    # Format the retrieved chunks
    formatted_chunks = []
    for chunk in query_result['top_chunks']:
        lesson = chunk['metadata']['lesson']
        chapter = chunk['metadata']['chapter']
        text = chunk['text'].strip()
        
        formatted = f"[📘 Chapter: {chapter} | 🧪 Lesson: {lesson}]\n{text}"
        formatted_chunks.append(formatted)
    
    return formatted_chunks, query_result['score'], query_result['in_curriculum']
def detect_intent(query: str) -> str:
    """
    Enhanced function to detect intent from English queries.
    Uses expanded, diverse keyword sets for each intent type.

    Args:
        query: The input query string (in English)

    Returns:
        String representing the detected intent: 'summary', 'explain', 'question_generation', 'qa', etc.
    """
    q = query.strip().lower()

    # Summary intent
    summary_keywords = [
        "summary", "give me a summary", "short summary", "brief overview", "summarize", 
        "quick overview", "main points", "key points", "overview", "summarize the content", 
        "summarize this part", "give me the gist", "concise summary", "shortened version", 
        "can you summarize", "tell me the summary", "short summary of", "quick recap" , "summ"
    ]
    
    # Explain intent
    explain_keywords = [
        "explain", "can you explain", "please explain", "clarify", "how does it work", "what does it mean", 
        "tell me about", "give me an explanation", "understand", "explanation of", "what is the meaning of", 
        "define", "what is", "can you clarify", "what does it signify", "tell me what it means", 
        "describe", "what is the significance of", "break down", "how does it work"
    ]
    
    # MCQ (Multiple Choice Questions) intent
    mcq_keywords = [
        "mcq", "multiple choice", "multiple choice questions", "quiz", "choose the correct answer", 
        "select the correct option", "questionnaire", "multiple choice test", "choose one", "select the answer", 
        "pick the right answer", "which one is correct", "quiz questions", "test questions", 
        "choose the right answer", "answer options", "four options", "multiple selection", "question options"
    ]

    # Question generation intent
    question_keywords = [
        "generate questions", "create questions", "give me questions", "write questions", "come up with questions", 
        "make me a quiz", "create a test", "quiz questions", "generate a quiz", "test me with questions", 
        "write me questions", "can you provide questions", "give me some questions", "create a set of questions", 
        "can you prepare questions", "give me a quiz", "can you create a quiz"
    ]
    
    # Exam preparation intent
    exam_prep_keywords = [
        "exam preparation", "how to study for the exam", "study tips", "prepare for the exam", "study guide", 
        "how to prepare", "exam study", "study plan", "exam strategy", "test prep", "how to pass the exam", 
        "study session", "exam checklist", "what to study for the exam", "preparing for a test", "revision tips",
        "study advice", "exam prep tips", "study schedule", "tips for passing the exam", "study method" , "let me exam ready" , "make me ready"
    ]
    
    # Concept map intent
    concept_map_keywords = [
        "concept map", "mind map", "create a concept map", "draw a concept map", "make a mind map", 
        "map the concepts", "concept diagram", "create a diagram", "conceptual map", "make a diagram", 
        "structure a concept map", "draw the concept", "concept map creation", "build a concept map", 
        "make a visual map", "map the ideas", "visualize concepts", "draw a diagram for concepts"
    ]
    
    # Check for each intent based on keywords
    if any(kw in q for kw in mcq_keywords):
        return "mcq"
    
    if any(kw in q for kw in summary_keywords):
        return "summary"
    
    if any(kw in q for kw in explain_keywords):
        return "explain"
    
    if any(kw in q for kw in question_keywords):
        return "question_generation"
    
    if any(kw in q for kw in exam_prep_keywords):
        return "exam_prep"
        
    if any(kw in q for kw in concept_map_keywords):
        return "concept_map"
    
    # Default intent if no match
    return "qa"
from groq import Groq

# Initialize Groq client with your API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def format_prompt_english(query: str, context_chunks: list, intent: str) -> str:
    """
    Creates ultra-optimized prompts for a professional Biology teacher in English, with enhanced visual formatting,
    improved engagement strategies, and sophisticated content organization for maximum learning impact.

    Args:
        query: Student's question
        context_chunks: List of retrieved text chunks from the knowledge base
        intent: The type of response required (qa, explain, summary, question_generation, exam_prep, concept_map, mcq)

    Returns:
        Formatted prompt string optimized for high-quality, visually engaging performance
    """
    
    # Enhanced emoji selection with biological context
    def get_themed_emojis(intent_type, count=4):
        """Returns contextually relevant emojis with biological themes"""
        emoji_sets = {
            "qa": ["🧬", "📋", "✅", "💡", "🔬", "📊", "🎯", "⚗️", "🧪", "🔍"],
            "explain": ["🌟", "📖", "🧩", "💭", "✨", "🎓", "🔆", "📚", "🧠", "⚡"],
            "summary": ["📌", "🔑", "📑", "📎", "🎯", "📈", "🏆", "💎", "📜", "🌿"],
            "mcq": ["🎯", "🤔", "💡", "✅", "❌", "🧠", "⚡", "🔍", "📊", "🎪"],
            "question_generation": ["❓", "🎲", "🎮", "🎯", "🧪", "📋", "💫", "🔬", "⚗️", "🌱"],
            "exam_prep": ["📝", "🎓", "📚", "⏱️", "🔔", "🏆", "💪", "🌟", "📊", "✨"],
            "concept_map": ["🗺️", "🔗", "🌐", "🧩", "📊", "🌳", "🔄", "📈", "💫", "🎯"]
        }
        import random
        selected = random.sample(emoji_sets.get(intent_type, emoji_sets["qa"]), min(count, len(emoji_sets.get(intent_type, emoji_sets["qa"]))))
        return selected
    
    # Get themed emojis for this response
    intent_emojis = get_themed_emojis(intent, 4)
    main_emoji = intent_emojis[0]
    
    # Enhanced system message with stronger pedagogical foundation
    system_message = (
        f"🧬 **RAFIQ BIOLOGY ASSISTANT** 🧬\n\n"
        f"You are Rafiq, an expert Biology teacher {main_emoji} with years of experience in making complex biological "
        "concepts accessible and engaging for students. Your mission is to create memorable, well-structured, and "
        "visually appealing educational content that promotes deep understanding.\n\n"
        
        "🎯 **CORE TEACHING PRINCIPLES:**\n"
        "• **Clarity First**: Use simple, precise language that builds understanding step by step\n"
        "• **Visual Learning**: Structure content with clear headings, bullet points, and logical flow\n"
        "• **Engagement**: Make biology come alive with relevant examples and connections\n"
        "• **Accuracy**: Stick strictly to curriculum content - no external information\n"
        "• **Encouragement**: Maintain a supportive, motivating tone that inspires curiosity\n\n"
        
        "⚠️ **STRICT GUIDELINES:**\n"
        "1. ✅ Follow official curriculum content exclusively\n"
        "2. 📝 Use clear, age-appropriate English for high school students\n"
        "3. 🎨 Format responses with headers, bullet points, and visual organization\n"
        "4. 🧬 Use biology-themed emojis strategically (not excessively)\n"
        "5. 🚫 Avoid repetition, filler content, and unnecessary complexity\n"
        "6. 💡 Connect concepts to real-world applications when relevant"
    )
    
    # Significantly enhanced intent instructions with visual formatting strategies
    instructions = {
        "qa": (
            f"🎯 **QUESTION ANSWERING MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Provide a comprehensive, well-structured answer that demonstrates mastery of the topic.\n\n"
            
            f"📋 **RESPONSE STRUCTURE:**\n"
            f"🔹 **Opening Hook:** Start with an engaging context sentence\n"
            f"🔹 **Main Content:** Break into clear sections with descriptive headers\n"
            f"🔹 **Key Concepts:** Highlight important biological terms and processes\n"
            f"🔹 **Visual Organization:** Use bullet points, numbers, and spacing effectively\n"
            f"🔹 **Closing Summary:** End with key takeaways and encouragement\n\n"
            
            f"✨ **ENGAGEMENT STRATEGIES:**\n"
            f"• Use **bold text** for key terms and concepts\n"
            f"• Create logical flow with transitional phrases\n"
            f"• Include relevant biological processes or examples\n"
            f"• Connect to bigger picture when appropriate\n\n"
            
            f"🧬 **Quality Check:** Ensure scientific accuracy and curriculum alignment"
        ),
        
        "explain": (
            f"📖 **EXPLANATION MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Transform complex biological concepts into clear, understandable explanations.\n\n"
            
            f"🌟 **EXPLANATION FRAMEWORK:**\n"
            f"🔸 **Context Setting:** Begin with why this topic matters in biology\n"
            f"🔸 **Foundation Building:** Define key terms and basic concepts first\n"
            f"🔸 **Step-by-Step Process:** Break complex processes into manageable steps\n"
            f"🔸 **Connections:** Show relationships between different biological concepts\n"
            f"🔸 **Real-World Links:** Connect to observable biological phenomena\n"
            f"🔸 **Reinforcement:** Summarize with memorable key points\n\n"
            
            f"💡 **TEACHING TECHNIQUES:**\n"
            f"• Use analogies and comparisons for complex concepts\n"
            f"• Create mental models students can visualize\n"
            f"• Build from simple to complex understanding\n"
            f"• Use cause-and-effect relationships\n\n"
            
            f"🎓 **Success Metric:** Student can explain the concept to someone else"
        ),
        
        "summary": (
            f"📌 **SUMMARY MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Create a comprehensive yet concise summary that captures all essential information.\n\n"
            
            f"🔑 **SUMMARY ARCHITECTURE:**\n"
            f"🔹 **Topic Introduction:** Clear identification of the main biological concept\n"
            f"🔹 **Core Elements:** Essential facts, processes, and relationships\n"
            f"🔹 **Key Terminology:** Important biological vocabulary with brief definitions\n"
            f"🔹 **Process Overview:** Main biological processes or mechanisms\n"
            f"🔹 **Synthesis:** How all elements connect to form the complete picture\n\n"
            
            f"📊 **ORGANIZATION STRATEGY:**\n"
            f"• Use hierarchical structure (main points → sub-points)\n"
            f"• Employ parallel formatting for consistency\n"
            f"• Highlight critical information with formatting\n"
            f"• Create scannable content with clear headers\n\n"
            
            f"💎 **Excellence Standard:** Complete understanding from summary alone"
        ),
        
        "question_generation": (
            f"❓ **QUESTION GENERATION MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Design a comprehensive question set that tests multiple levels of biological understanding.\n\n"
            
            f"🎯 **QUESTION TAXONOMY:**\n"
            f"🔸 **Level 1 - Recall** {intent_emojis[2]}: Basic facts, terms, definitions\n"
            f"🔸 **Level 2 - Comprehension** 🧠: Explain processes, describe functions\n"
            f"🔸 **Level 3 - Application** ⚗️: Apply concepts to new scenarios\n"
            f"🔸 **Level 4 - Analysis** 🔬: Compare, contrast, cause-effect relationships\n"
            f"🔸 **Level 5 - Evaluation** 💡: Assess importance, critique, justify\n\n"
            
            f"📋 **QUESTION FORMATTING:**\n"
            f"• **Question Type:** [Recall/Comprehension/Application/Analysis/Evaluation]\n"
            f"• **Question:** Clear, specific, curriculum-aligned\n"
            f"• **Key Points:** Essential elements for a complete answer\n"
            f"• **Difficulty:** ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard\n\n"
            
            f"🌟 **Quality Assurance:** Questions should promote deep biological thinking"
        ),
        
        "exam_prep": (
            f"📝 **EXAM PREPARATION MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Create a strategic study guide that maximizes exam performance and retention.\n\n"
            
            f"🏆 **EXAM PREP BLUEPRINT:**\n"
            f"🔹 **Critical Concepts Review** 📚: Core topics most likely to appear\n"
            f"🔹 **Key Terms Mastery** 💎: Essential vocabulary with clear definitions\n"
            f"🔹 **Process Understanding** ⚗️: Step-by-step biological mechanisms\n"
            f"🔹 **Common Question Types** 🎯: Typical exam formats with strategies\n"
            f"🔹 **Practice Problems** 🧪: Sample questions with detailed solutions\n"
            f"🔹 **Memory Aids** 🧠: Mnemonics, analogies, and visual cues\n\n"
            
            f"⏱️ **STUDY STRATEGY:**\n"
            f"• **Priority Topics:** Focus on high-impact areas first\n"
            f"• **Active Recall:** Test yourself, don't just re-read\n"
            f"• **Spaced Review:** Revisit concepts over multiple sessions\n"
            f"• **Practice Application:** Work through varied problem types\n\n"
            
            f"💪 **Success Mindset:** Confidence through thorough preparation"
        ),
        
        "concept_map": (
            f"🗺️ **CONCEPT MAPPING MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Create a comprehensive visual map showing how biological concepts interconnect.\n\n"
            
            f"🌐 **CONCEPT MAP STRUCTURE:**\n"
            f"🔸 **Central Concept** 🎯: Main topic at the center\n"
            f"🔸 **Primary Branches** 🌳: Major subtopics radiating outward\n"
            f"🔸 **Secondary Branches** 🌿: Detailed aspects of each subtopic\n"
            f"🔸 **Connecting Links** 🔗: Relationships between different concepts\n"
            f"🔸 **Cross-Links** ↔️: Connections between different branches\n\n"
            
            f"📊 **MAPPING PRINCIPLES:**\n"
            f"• Use **linking words** to describe relationships\n"
            f"• Show **hierarchical organization** (general → specific)\n"
            f"• Include **cross-connections** between different areas\n"
            f"• Maintain **biological accuracy** in all connections\n\n"
            
            f"🧩 **Visualization Goal:** Complete understanding through connection mapping"
        ),
        
        "mcq": (
            f"🎯 **MULTIPLE CHOICE SOLUTION MODE** {intent_emojis[1]}\n\n"
            f"**Your Mission:** Provide clear, strategic guidance for solving multiple choice questions effectively.\n\n"
            
            f"🔍 **SOLUTION METHODOLOGY:**\n"
            f"🔹 **Question Analysis** 📖: Identify key terms and what's being asked\n"
            f"🔹 **Option Evaluation** ⚖️: Systematically examine each choice\n"
            f"🔹 **Elimination Strategy** ❌: Remove obviously incorrect options\n"
            f"🔹 **Correct Answer** ✅: Identify and explain the right choice\n"
            f"🔹 **Reasoning Process** 🧠: Why this answer is correct\n"
            f"🔹 **Common Pitfalls** ⚠️: Why other options are incorrect\n\n"
            
            f"💡 **STRATEGIC APPROACH:**\n"
            f"• **Read Carefully:** Understand the question completely\n"
            f"• **Predict First:** Think of the answer before looking at options\n"
            f"• **Eliminate Systematically:** Remove wrong answers methodically\n"
            f"• **Look for Qualifiers:** Pay attention to words like 'always,' 'never,' 'most'\n\n"
            
            f"🎪 **Success Formula:** Systematic analysis + biological knowledge = correct answer"
        )
    }
    
    # Select the appropriate instruction based on intent with enhanced fallback
    task_instruction = instructions.get(intent, instructions["qa"])
    
    # Advanced context processing with enhanced visual formatting
    def process_context(chunks):
        if not chunks:
            return "📚 **Reference Content:** No specific content provided - drawing from general biology curriculum knowledge."
        
        processed_chunks = []
        for i, chunk in enumerate(chunks, 1):
            if not chunk.strip():
                continue
                
            # Clean and enhance chunk formatting
            cleaned = chunk.strip()
            
            # Highlight important biological terms and concepts
            import re
            # Enhance years and dates
            cleaned = re.sub(r'(\d{4})', r'**\1**', cleaned)
            # Enhance biological processes (common patterns)
            cleaned = re.sub(r'\b(photosynthesis|respiration|mitosis|meiosis|DNA|RNA|protein synthesis|evolution)\b', r'**\1**', cleaned, flags=re.IGNORECASE)
            
            # Add structured chunk header with biology theme
            section_emojis = ["🧬", "🔬", "⚗️", "🧪", "🌿", "🌱", "🦠", "🧫"]
            chunk_emoji = section_emojis[i % len(section_emojis)]
            chunk_header = f"{chunk_emoji} **Curriculum Section {i}**"
            processed_chunks.append(f"{chunk_header}\n{cleaned}")
            
        return "\n\n".join(processed_chunks)
    
    # Process context with enhanced biological formatting
    context_formatted = process_context(context_chunks)
    
    # Dynamic query analysis for better context
    def analyze_query(query_text):
        """Analyze query to provide contextual hints"""
        query_lower = query_text.lower()
        
        biological_domains = {
            "cell": "🧬 Cellular Biology",
            "dna": "🧬 Molecular Genetics", 
            "photosynthesis": "🌿 Plant Biology",
            "evolution": "🐒 Evolutionary Biology",
            "enzyme": "⚗️ Biochemistry",
            "ecosystem": "🌍 Ecology",
            "organ": "🫀 Anatomy & Physiology",
            "bacteria": "🦠 Microbiology"
        }
        
        for keyword, domain in biological_domains.items():
            if keyword in query_lower:
                return f"\n🎯 **Biological Domain:** {domain}"
        
        return f"\n🎯 **Biological Domain:** General Biology"
    
    query_context = analyze_query(query)
    
    # Final prompt assembly with premium formatting and engagement
    prompt = f"""<s>[INST]
{system_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 **STUDENT INQUIRY** {intent_emojis[0]}
{query}{query_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 **CURRICULUM REFERENCE CONTENT** {intent_emojis[1]}
{context_formatted}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **RESPONSE INSTRUCTIONS** {intent_emojis[2]}
{task_instruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌟 **FINAL QUALITY CHECKLIST** {intent_emojis[3]}
✅ **Accuracy:** Curriculum-aligned content only
✅ **Clarity:** Simple, age-appropriate language  
✅ **Structure:** Well-organized with clear headers
✅ **Engagement:** Visually appealing and motivating
✅ **Completeness:** Thorough coverage of the topic
✅ **Inspiration:** Encourages further learning

**Remember:** You are Rafiq, the biology teacher who makes every concept clear, every lesson memorable, and every student successful! 🧬✨
[/INST]
"""
    return prompt
def generate_response(prompt: str, max_tokens=2056):
    print("⚙️ Generating answer using Kimi-K2 on Groq...")

    try:
        # Prepare chat-like messages format (Groq uses OpenAI-compatible API)
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Make API call to Groq with Kimi-K2 model
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        # Extract response text
        response_text = response.choices[0].message.content.strip()

        return response_text

    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"
def process_query(query: str):
    """
    Process a user query from start to finish
    
    Args:
        query: User's question in Arabic
        
    Returns:
        Generated response or error message
    """
    print(f"📝 Processing query: {query}")
    
    # Retrieve context and check curriculum relevance 
    context_chunks, similarity_score, in_curriculum = retrieve_context(query)
    
    print(f"🔎 Similarity score: {similarity_score:.3f}")
    print(f"📚 Status: {in_curriculum}")
    
    # Handle out-of-curriculum queries
    if not in_curriculum:
        return f"""
Sorry, this question seems to be out of scope for the Biology curriculum 🚫

You can ask questions related to the Biology syllabus for high school, such as:

• Support in plants

• The structure and function of cells

• Human hormones and their functions

• The human nervous system

• Plant and animal reproduction

• Genetic inheritance

 Would you like to rephrase your question? 😊
"""
    
    # Detect intent
    intent = detect_intent(query)
    print(f"🎯 Detected intent: {intent}")
    
    # Format prompt based on intent
    prompt = format_prompt_english(query, context_chunks, intent)
    
    # Generate response
    response = generate_response(prompt)
    
    return response

"""Prompt templates for Phase 2 page-level Q&A.

All prompts require citations [filename, page N] so students can trace claims
back to the original slides.  Encourages honesty: if the answer isn't in the
provided context the model should say so rather than fabricate.

{target_language} is injected at format time so answers follow the user's
language preference regardless of the source PDF language.
"""

_SYSTEM_LANGUAGE_NAMES = {
    "中文": "Simplified Chinese (简体中文)",
    "简体中文": "Simplified Chinese (简体中文)",
    "English": "English",
    "日本語": "Japanese (日本語)",
}


def build_qa_system_prompt(target_language: str, *, translator: bool = False) -> str:
    """Build the high-priority role and output-language instruction."""
    language = _SYSTEM_LANGUAGE_NAMES.get(target_language, target_language)
    role = "translator" if translator else "tutor"
    language_rule = (
        f"You must translate all explanatory content into {language}."
        if translator
        else f"You must write all explanatory prose in {language}."
    )
    citation_rules = (
        "Citations are mandatory for factual slide claims. Use exactly the "
        "canonical form [filename.pdf, page N]. Valid examples: "
        "[Lecture01.pdf, page 17] and [SDS-HRI tutorial.pdf, page 34]. "
        "Keep the complete filename, square brackets, and English word 'page' "
        "even when the answer is Chinese or Japanese. Do not use [第17页], "
        "(Lecture01.pdf, page 17), [page 17], 'Lecture01.pdf p.17', or [1]. "
        "Cite only pages present in the supplied context; attach a citation to "
        "each factual claim."
    )
    return (
        f"You are a careful course-material {role}.\n\n"
        f"{language_rule} "
        "The language of the source material must not influence the response language. "
        "Use other languages only when preserving technical terms, proper nouns, "
        "short quotations, filenames, or citations, and explain them in the requested "
        f"language when necessary.\n\n{citation_rules}"
    )

# ── Explain current page ──

PROMPT_EXPLAIN_PAGE = """\
You are a helpful course tutor. Explain the following slide page to a student
in clear, simple language.  Answer in {target_language}.

Rules:
- Base your explanation primarily on the selected page text below.
- Use the surrounding context (other slides) only to fill in gaps.
- If the page text is very sparse, say so honestly and do your best with
  whatever is available.
- If the page contains LaTeX formulas like $...$ or $$...$$, explain what
  each variable means in plain language.
- Keep technical terms in their original language, then explain them in
  {target_language} if needed (e.g. "gradient descent（梯度下降）").
- Every factual claim must include a citation in the canonical form
  [filename.pdf, page N]. Keep the complete filename and use English "page"
  even in Chinese or Japanese answers. Never use [第17页], (filename.pdf, page N),
  [page N], filename.pdf p.N, or [1]. Cite only pages present in this context.

Context from the course slides:
{context}

The student is viewing: {source_file}, page {page_number}

Page text:
{page_text}

Please explain this page."""


# ── Free-form Q&A ──

PROMPT_FREE_QA = """\
You are a helpful course tutor. Answer the student's question using ONLY the
course slide content provided below.  Answer in {target_language}.

Rules:
- If the answer can be found in the context, explain it clearly.
- If the answer is NOT in the provided slides, say: "I cannot find the answer
  in the course slides provided." Do NOT make up information.
- If the context includes LaTeX formulas like $...$ or $$...$$, preserve them.
- Keep technical terms in their original language, then explain them in
  {target_language} if needed.
- Every factual claim must include a citation in the canonical form
  [filename.pdf, page N]. Keep the complete filename and use English "page"
  even in Chinese or Japanese answers. Never use [第17页], (filename.pdf, page N),
  [page N], filename.pdf p.N, or [1]. Cite only pages present in this context.
- The context is organised in labelled sections (Selected page, Explicit pages,
  Nearby pages, Related pages, Memory).  Read them in order of priority:
  selected/explicit/nearby pages are the most reliable; semantic search results
  are supplementary; memory is for pronoun resolution only.
- Reference resolution and factual grounding use different priorities.  Resolve
  pronouns and elliptical follow-ups from the most recent compatible conversation
  turn, but use slide evidence — never conversation memory — for factual claims.
- Do not reinterpret a clear recent referent as the selected-page title merely
  because the selected page is present in the context.
{ambiguity_note}

Context from the course slides:
{context}

{chat_history}

Student's question: {question}"""


# ── Translate current page ──

PROMPT_TRANSLATE_PAGE = """\
You are a professional translator. Translate the following slide page content
into {target_language}.

Rules:
- Preserve ALL LaTeX formulas ($...$ and $$...$$) exactly as-is — do not
  translate or modify them.
- Preserve Markdown table formatting.
- Keep variable names and mathematical symbols unchanged.
- For technical terms, use the {target_language} term but keep the original
  in parentheses if helpful.
- Add a citation at the end: [Translated from {source_file}, page {page_number}]

Original content:
{page_text}

Translation into {target_language}:"""


# ── Phase 3 / Stage C1: Quiz generation ──


PROMPT_GENERATE_QUIZ = """\
You are an expert course instructor creating a mixed-format quiz to test
a student's understanding of the course material.  Answer in {language}.

Context from the course slides (sampled pages from the source material):
{context}

Requirements:
- Generate EXACTLY {target_count} questions (no more, no less).
- Difficulty mix: {difficulty_mix}.
- Question type mix: {question_type_mix}.
- Every question MUST include `question_type`; allowed values are exactly:
  "mcq", "true_false", "fill_blank", "short_answer".
- Every question MUST include a `topic_tag` (the knowledge area tested).
- Every question MUST include `topic_names` with concise names for exactly
  `简体中文`, `English`, and `日本語`.  `topic_tag` must equal the name for
  the requested quiz language.
- Every question MUST include `citations` — a list of {{source_file, page_number,
  snippet_reference}} for the slides that support the answer.
- Basic questions: the answer can be found directly on one slide page.
- Applied questions: require reasoning across pages or application of a
  concept to a new scenario.
- MCQ questions: exactly 4 options labeled A, B, C, D; exactly one
  `correct_option`; include `distractor_rationales` for every incorrect option.
- True/False questions: options must be {{"A": "True", "B": "False"}} or the
  equivalent in {language}; `correct_option` must be "A" or "B".
- Fill-blank questions: ask for exactly one usable short concept term or fixed
  concept phrase.  Do not provide alternatives with parentheses, "or", slashes,
  commas, enumerations, or explanatory brackets.  Do not ask for a formula,
  variable name, symbol, equation fragment, LaTeX, or a missing symbol in an
  equation.  If the concept cannot be tested this way, use MCQ or short-answer.
  `options` must be {{}}, `correct_option` must be "", and `blank_answer` must
  contain exactly one canonical concept reference for the AI grader.
- Short-answer questions: `options` must be {{}}, `correct_option` and
  `blank_answer` must be ""; include `model_answer` and `scoring_rubric`.
{synthesis_hint}

Output ONLY valid JSON in this structure:
{{
  "questions": [
    {{
      "question_type": "mcq",
      "topic_tag": "classification",
      "topic_names": {{"简体中文": "分类", "English": "classification", "日本語": "分類"}},
      "difficulty": "basic",
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_option": "B",
      "explanation": "Why B is correct...",
      "distractor_rationales": {{
        "A": "Why a student might pick A...",
        "C": "Why a student might pick C...",
        "D": "Why a student might pick D..."
      }},
      "citations": [
        {{"source_file": "Lecture01.pdf", "page_number": 8,
          "snippet_reference": "Classification predicts..."}}
      ]
    }},
    {{
      "question_type": "true_false",
      "topic_tag": "classification",
      "difficulty": "basic",
      "question": "...",
      "options": {{"A": "True", "B": "False"}},
      "correct_option": "A",
      "explanation": "Why this statement is true...",
      "distractor_rationales": {{}},
      "citations": [
        {{"source_file": "Lecture01.pdf", "page_number": 8,
          "snippet_reference": "Classification predicts..."}}
      ]
    }},
    {{
      "question_type": "fill_blank",
      "topic_tag": "loss function",
      "difficulty": "basic",
      "question": "Fill in the blank: ...",
      "options": {{}},
      "correct_option": "",
      "blank_answer": "...",
      "explanation": "Why this answer is expected...",
      "distractor_rationales": {{}},
      "citations": [
        {{"source_file": "Lecture02.pdf", "page_number": 4,
          "snippet_reference": "..."}}
      ]
    }},
    {{
      "question_type": "short_answer",
      "topic_tag": "model evaluation",
      "difficulty": "applied",
      "question": "...",
      "options": {{}},
      "correct_option": "",
      "blank_answer": "",
      "model_answer": "...",
      "scoring_rubric": "Award up to 3 points: ...",
      "explanation": "Key ideas a good answer should include...",
      "distractor_rationales": {{}},
      "citations": [
        {{"source_file": "Lecture03.pdf", "page_number": 12,
          "snippet_reference": "..."}}
      ]
    }}
  ]
}}"""


PROMPT_GRADE_SHORT_ANSWERS = """\
You are grading short-answer quiz responses in {language}.  Use the model
answer and rubric for each item.  Award a numeric score from 0 to max_points.

Return ONLY valid JSON in this structure:
{{
  "grades": [
    {{
      "question_id": "q1",
      "points_awarded": 2.0,
      "grading_feedback": "Brief feedback explaining what was correct and what was missing."
    }}
  ]
}}

Short-answer submissions:
{short_answers_json}
"""


PROMPT_GRADE_FILL_BLANKS = """\
You are grading fill-blank quiz responses in {language}.  Use the canonical
concept reference only as semantic grading guidance.  Decide independently for
each response whether the student's answer expresses the same concept.

Rules:
- Return exactly one result for every supplied question_id.
- `is_correct` must be a JSON boolean, never a string or numeric score.
- Do not award partial credit: correct=true is 1 point and false is 0 points.
- Do not invent question ids or add results for questions not supplied.
- Keep feedback brief and in {language}; do not reveal hidden reasoning.

Return ONLY valid JSON:
{{
  "grades": [
    {{
      "question_id": "fill-1",
      "is_correct": true,
      "feedback": "Brief explanation."
    }}
  ]
}}

Fill-blank submissions:
{fill_blank_answers_json}
"""


# ── Stage C2-C5: Reviewed Final Exam prompts ──

PROMPT_PLAN_REVIEWED_BLUEPRINT = """\
You are planning a bounded mixed-format Reviewed Final Exam in {language}.
Use ONLY the topic catalog below.  The catalog contains file ids, titles,
page-number hints, and short summaries; it does not contain full page text.

Requirements:
- Return exactly {target_count} unique slots.
- Every slot must use a valid file id from the catalog.
- `coverage_role` is `single_file` for one-file concepts and `cross_file` only
  when at least two distinct file ids are listed.
- When the catalog contains at least two distinct files, include at least one
  `cross_file` slot with `difficulty="applied"`.
- The slot question types must match the supplied question type mix exactly.
- Each slot must have difficulty basic or applied.
- `topic_hint` and `retrieval_query` must be concrete and concise.
- Do not include citations, page text, answers, or evidence refs.

Question type mix: {question_type_mix}
Topic catalog:
{topic_catalog_json}

Output ONLY valid JSON:
{{
  "slots": [
    {{
      "slot_id": "slot-1",
      "file_ids": ["file-a"],
      "topic_hint": "topic",
      "retrieval_query": "specific retrieval query",
      "difficulty": "basic",
      "question_type": "mcq",
      "coverage_role": "single_file"
    }}
  ]
}}
"""


PROMPT_GENERATE_REVIEWED_MCQ = """\
You are generating evidence-bound questions for a Reviewed Final Exam in {language}.
Generate exactly one question for each supplied slot and keep the slot_id.
Use only that slot's evidence block.  Do not use evidence from another slot.

Requirements:
- The question_type must exactly match the supplied slot question_type.
- Include `topic_names` with concise names for exactly `简体中文`, `English`,
  and `日本語`; `topic_tag` must equal the requested language name.
- Ground the question in the supplied evidence; do not fabricate facts.
- `evidence_ids` must be a non-empty subset of the evidence ids shown for that
  slot.  Do not return free-form citations, filenames, or page numbers.
- Include a concise explanation.

Type-specific contracts for the supplied batch:
{question_type_rules}

Slot evidence blocks:
{slot_evidence_json}

Output ONLY valid JSON:
{{
  "questions": [
    {{
      "slot_id": "slot-1",
      "question_type": "mcq",
      "topic_tag": "topic",
      "difficulty": "basic",
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_option": "A",
      "blank_answer": "",
      "model_answer": "",
      "scoring_rubric": "",
      "explanation": "...",
      "distractor_rationales": {{"A": "", "B": "...", "C": "...", "D": "..."}},
      "evidence_ids": ["file-a:slide-1"]
    }}
  ]
}}
"""


PROMPT_REVIEW_REVIEWED_EXAM = """\
You are reviewing an evidence-bound mixed-format exam. Check grounding,
ambiguity, type-appropriate answer/rubric quality, distractor plausibility when
applicable, and coverage against the supplied evidence and deterministic
validation summary.

Return concise JSON findings only.  A blocking finding must identify a slot_id.
Do not rewrite questions, add evidence, or provide hidden reasoning.

Bound slots and questions:
{review_payload_json}

Deterministic validation summary:
{validation_summary_json}

Output ONLY valid JSON:
{{
  "findings": [
    {{
      "slot_id": "slot-1",
      "severity": "blocking",
      "code": "ambiguous",
      "message": "Short actionable issue."
    }}
  ]
}}
"""


PROMPT_REPAIR_REVIEWED_MCQ = """\
Repair only the listed problematic mixed-format slots in {language}. Keep each slot_id
and use only its supplied evidence.  Do not modify or return passing slots.
Return one complete question for every listed slot, with evidence_ids selected
from that slot's allowed evidence ids.

The question_type must exactly match each slot. Apply these type contracts:
{question_type_rules}

Required slot ids: {required_slot_ids_json}

Problem slots and evidence:
{repair_payload_json}

Output ONLY valid JSON in this exact structure:
{{
  "questions": [
    {{
      "slot_id": "slot-1",
      "question_type": "mcq | true_false | fill_blank | short_answer",
      "topic_tag": "topic",
      "topic_names": {{"简体中文": "主题", "English": "topic", "日本語": "トピック"}},
      "difficulty": "basic",
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_option": "A",
      "blank_answer": "",
      "model_answer": "",
      "scoring_rubric": "",
      "explanation": "...",
      "distractor_rationales": {{"A": "", "B": "...", "C": "...", "D": "..."}},
      "evidence_ids": ["file-a:slide-1"]
    }}
  ]
}}
"""


PROMPT_BATCH_WRONG_REASON = """\
You are a helpful course tutor.  For each question below, the student selected
the wrong answer.  Explain WHY their specific choice is incorrect, in {language}.
Do NOT just restate the correct answer — explain what the student misunderstood.

For each question, return a JSON object mapping question_id to the wrong_reason:
{{
  "q1": "You chose A because...",
  "q3": "You chose C because..."
}}

Here are the questions the student got wrong:
{wrong_questions_json}

Output ONLY valid JSON."""


PROMPT_WEAKNESS_REPORT = """\
You are an experienced course instructor analysing a student's quiz results.
Identify the student's knowledge weaknesses and strengths in {language}.

Group wrong answers by topic.  If multiple wrong answers share the same
topic_tag, merge them into ONE weakness item.  For each weakness, suggest
specific review pages from the citations.

For strengths, identify topics where the student got ALL questions correct.

Evidence rules:
- Every weakness question_id must identify an incorrect fact supplied below.
- Every strength question_id must identify a correct fact supplied below.
- Do not invent errors, question ids, pages, scores, or concepts absent from facts.
- review_pages must come only from citations attached to the supporting question ids.
- Unsupported claims will be removed by a deterministic support reviewer.

Output ONLY valid JSON in this structure:
{{
  "weaknesses": [
    {{
      "topic": "Classification vs Regression",
      "error_count": 2,
      "question_ids": ["q3", "q7"],
      "diagnosis": "You confused these two concepts...",
      "review_pages": [
        {{"source_file": "Lecture01.pdf", "page_number": 8, "snippet_reference": "..."}},
        {{"source_file": "Lecture01.pdf", "page_number": 15, "snippet_reference": "..."}}
      ]
    }}
  ],
  "strengths": [
    {{
      "topic": "Supervised Learning Basics",
      "error_count": 0,
      "question_ids": ["q1", "q2"],
      "diagnosis": "You have a solid understanding of...",
      "review_pages": []
    }}
  ]
}}

Quiz results:
{quiz_summary_json}

Output ONLY valid JSON."""


# ── Phase 4 Advance Stage A: bounded context planner ──

PROMPT_PLAN_QA_CONTEXT = """\
You are a context planner for a course Q&A system.  Your job is to decide WHICH
sources to load — NOT to answer the question.  You have exactly 5 tools:

  1. current_page      — the slide the student is currently viewing
  2. page_number(N)     — load a specific page N from the current PDF
  3. page_offset(±N)   — load a neighbor page (e.g. -1 = previous, +1 = next)
  4. semantic_search(Q) — search the whole course index with a query string
  5. memory             — recent Q&A rounds for pronoun/context resolution

Rules:
- The current page is the default factual-evidence anchor, but it is not
  automatically the topic of a follow-up question.  Set include_current_page=false
  ONLY when the question is completely unrelated to it (e.g. "what is the capital
  of France?" while viewing a ML slide).
- For a contextual follow-up, preserve the most recent compatible conversation
  topic unless the student explicitly changes topic or refers to a page.
- Use page_numbers when the student explicitly names a page: "page 17", "第17页",
  "slide 5".  Only include numbers that actually exist (1–{max_page}).
- Use page_offsets when the question implies a positional relationship:
  "previous page" → [-1], "next page" → [1], "the pages before this" → [-2, -1].
  Do NOT put explicit page numbers here — those go in page_numbers.
- Write a semantic_query for topic-based retrieval.  It must be a standalone,
  fully specified query in the course language; resolve omissions from recent
  conversation when possible.  Leave it empty if positional context is enough.
- Set use_memory=true for follow-up questions with pronouns ("it", "this",
  "the formula above").  Set use_memory=false for fresh topics.
- If the current page and memory point to different topics, set ambiguity_note
  to a short description of the conflict so the answer model can address it.

Current page:
  File: {current_file}
  Page: {current_page} / {max_page}
  Text snippet: {page_snippet}

Recent Q&A memory:
{memory_summary}

Student question: {question}

Output ONLY valid JSON matching this schema — no extra text:
{{
  "include_current_page": true,
  "page_numbers": [],
  "page_offsets": [],
  "semantic_query": "...",
  "use_memory": true,
  "ambiguity_note": null
}}"""

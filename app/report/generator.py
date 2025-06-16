# Générateur de rapports avec mémoire
import os
import json
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI

from app.report.prompts import (
    STRUCTURE_PROMPT,
    SECTION_ANALYSIS_PROMPT,
    KEY_POINTS_PROMPT,
    FINAL_REPORT_PROMPT
)

class ReportGenerator:
    def __init__(self, vector_db, llm_model=None):
        self.vector_db = vector_db

        if llm_model is None:
            self.llm = GoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.2
            )
        else:
            self.llm = llm_model

        self.context_memory = {
            "document_structure": None,
            "key_findings": [],
            "section_summaries": {}
        }

    def extract_document_structure(self):
        samples = self.vector_db.db.similarity_search(
            "table des matières sommaire plan document",
            k=5
        )
        sample_texts = [doc.page_content for doc in samples]
        chain = LLMChain(llm=self.llm, prompt=STRUCTURE_PROMPT)
        result = chain.run(document_samples="\n\n".join(sample_texts))

        try:
            self.context_memory["document_structure"] = json.loads(result)
        except:
            self.context_memory["document_structure"] = {
                "title": "Document Analysis",
                "chapters": ["Introduction", "Content", "Conclusion"]
            }

        return self.context_memory["document_structure"]

    def analyze_section(self, section_title, section_query=None):
        if section_query is None:
            section_query = section_title

        results = self.vector_db.hybrid_search(query=section_query, k=8)
        previous_context = ""
        if self.context_memory["key_findings"]:
            previous_context = "Points clés identifiés jusqu'à présent:\n" + "\n".join(
                f"- {pt}" for pt in self.context_memory["key_findings"]
            )

        chain = LLMChain(llm=self.llm, prompt=SECTION_ANALYSIS_PROMPT)
        section_content = "\n\n".join([doc.page_content for doc in results])

        section_analysis = chain.run(
            section_title=section_title,
            previous_context=previous_context,
            section_content=section_content
        )
        self.context_memory["section_summaries"][section_title] = section_analysis

        key_points_chain = LLMChain(llm=self.llm, prompt=KEY_POINTS_PROMPT)
        key_points = key_points_chain.run(section_analysis=section_analysis)

        for point in key_points.strip().split("\n"):
            if point and point not in self.context_memory["key_findings"]:
                self.context_memory["key_findings"].append(point.strip())

        return section_analysis

    def generate_full_report(self):
        if not self.context_memory["document_structure"]:
            self.extract_document_structure()

        structure = self.context_memory["document_structure"]

        for chapter in tqdm(structure.get("chapters", [])):
            if isinstance(chapter, dict):
                chapter_title = chapter.get("title", "")
                if chapter_title and chapter_title not in self.context_memory["section_summaries"]:
                    self.analyze_section(chapter_title)

                for section in chapter.get("sections", []):
                    section_title = section if isinstance(section, str) else section.get("title", "")
                    if section_title and section_title not in self.context_memory["section_summaries"]:
                        self.analyze_section(section_title)
            else:
                if chapter not in self.context_memory["section_summaries"]:
                    self.analyze_section(chapter)

        document_title = structure.get("title", "Analyse de Document")
        document_structure = json.dumps(structure, indent=2)
        key_findings = "\n".join([f"- {pt}" for pt in self.context_memory["key_findings"]])

        section_summaries_text = ""
        for title, summary in self.context_memory["section_summaries"].items():
            section_summaries_text += f"### {title}\n{summary}\n\n"

        chain = LLMChain(llm=self.llm, prompt=FINAL_REPORT_PROMPT)
        final_report = chain.run(
            document_title=document_title,
            document_structure=document_structure,
            key_findings=key_findings,
            section_summaries=section_summaries_text
        )

        return final_report

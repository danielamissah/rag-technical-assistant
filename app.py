"""
Hugging Face Spaces entry point.
Wraps the RAG pipeline in a Gradio UI with Inter font and
improved UX — clear button after receiving an answer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from src.agent.rag_agent import RAGAgent

agent = RAGAgent()

EXAMPLES = [
    "What are the main challenges in LiDAR and camera sensor fusion?",
    "How do transformer architectures improve object detection in autonomous vehicles?",
    "What methods are used for anomaly detection in automotive sensor data?",
    "How do autonomous vehicles handle adverse weather conditions like rain and fog?",
    "What role does reinforcement learning play in autonomous driving systems?",
    "How is semantic segmentation used in autonomous vehicle perception pipelines?",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

h1 { font-weight: 700 !important; font-size: 1.8rem !important; }
h3 { font-weight: 600 !important; color: #1e3a5f !important; }

#ask-btn {
    background: #2E75B6 !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
}

#ask-another-btn {
    background: #f0f4f8 !important;
    color: #2E75B6 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 2px solid #2E75B6 !important;
    font-size: 1rem !important;
}

#answer-box textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

#question-box textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}

.source-card {
    background: #f8fafc;
    border-left: 3px solid #2E75B6;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
}
"""


def answer_question(question: str):
    if not question.strip():
        return (
            "Please enter a question.",
            "",
            gr.update(visible=True),   # ask btn visible
            gr.update(visible=False),  # ask another btn hidden
        )

    result  = agent.ask(question)
    answer  = result["answer"]
    sources = result["sources"]
    latency = result["latency_ms"]

    sources_md = f"\n\n**{len(sources)} papers retrieved** — latency: {latency}ms\n\n"
    for i, s in enumerate(sources, 1):
        sources_md += (
            f"**[{i}] {s['title']}**  \n"
            f"*{s['authors']} \u00B7 {s['published']}*  \n"
            f"Relevance: {s['score']:.1%} \u00B7 "
            f"[PDF]({s['pdf_url']}) \u00B7 "
            f"[arXiv](https://arxiv.org/abs/{s['arxiv_id']})\n\n"
        )

    return (
        answer,
        sources_md,
        gr.update(visible=False),  # hide ask btn
        gr.update(visible=True),   # show ask another btn
    )


def reset():
    return (
        "",    # clear question
        "",    # clear answer
        "",    # clear sources
        gr.update(visible=True),   # show ask btn
        gr.update(visible=False),  # hide ask another btn
    )


with gr.Blocks(css=CSS, title="RAG Technical Assistant") as demo:

    gr.Markdown("""
    # RAG Technical Assistant
    ### Autonomous Driving & Sensor Systems Research

    Ask questions about autonomous driving research. The system retrieves relevant papers
    from **389 arXiv papers** and generates cited answers using a language model.

    **Topics:** LiDAR/camera fusion · Object detection · Anomaly detection · Transformers · Semantic segmentation
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g. How does LiDAR-camera fusion work in autonomous vehicles?",
                lines=3,
                elem_id="question-box",
            )

            with gr.Row():
                ask_btn = gr.Button(
                    "Ask",
                    variant="primary",
                    elem_id="ask-btn",
                    visible=True,
                )
                ask_another_btn = gr.Button(
                    "Ask Another Question",
                    elem_id="ask-another-btn",
                    visible=False,
                )

            gr.Examples(
                examples=EXAMPLES,
                inputs=question,
                label="Example Questions",
            )

        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Answer",
                lines=14,
                interactive=False,
                elem_id="answer-box",
            )
            sources_box = gr.Markdown(label="Sources")

    ask_btn.click(
        fn=answer_question,
        inputs=question,
        outputs=[answer_box, sources_box, ask_btn, ask_another_btn],
    )

    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=[answer_box, sources_box, ask_btn, ask_another_btn],
    )

    ask_another_btn.click(
        fn=reset,
        inputs=[],
        outputs=[question, answer_box, sources_box, ask_btn, ask_another_btn],
    )

    gr.Markdown("""
    ---
    **Stack:** arXiv API · ChromaDB · sentence-transformers · Ollama / Mistral-7B · FastAPI · Gradio  
    **GitHub:** [rag-technical-assistant](https://github.com/danielamissah/rag-technical-assistant)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
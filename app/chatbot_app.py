"""Run from the project root: python -m streamlit run app/chatbot_app.py."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import create_runtime  # noqa: E402 -- Streamlit needs the repository on sys.path.


def show_response(st, response):
    st.markdown(response['answer_text'])
    if response['insufficient_evidence']:
        st.caption('The documents do not fully support an answer to this question.')
    if response['sources']:
        with st.expander('Sources and evidence'):
            for source in response['sources']:
                label = source.get('source_filename') or source['doc_id']
                st.write(f"{label} · {', '.join(source['evidence_ids'])}")
                if source.get('section'):
                    st.caption(source['section'])
                if source.get('page'):
                    st.caption(f"Page {source['page']}")
            for item in response['evidence']:
                st.write(item['evidence_id'])
                st.text(item.get('text') or f"{item['subject']} {item['relation']} {item['object']}")
    with st.expander('Answer details'):
        st.caption(f"Confidence: {response['confidence']:.0%} · Retrieval retries: {response['retries_used']}")
        for warning in response['warnings']:
            st.caption(warning)


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title='AI Memory Graph', page_icon='🧠', layout='wide')
    st.title('AI Memory Graph')
    st.write('Upload documents, connect their facts, and ask questions with sources.')
    mode = st.sidebar.radio('Workspace', ['Offline demo', 'Neo4j + Chroma'])
    demo = mode == 'Offline demo'
    st.sidebar.caption('Demo uses keyword search and quoted passages.' if demo else
                       'Uses your database and models configured in .env.')
    if st.session_state.get('mode') != mode:
        old = st.session_state.pop('runtime', None)
        if old:
            old.close()
        st.session_state['mode'] = mode
        st.session_state['messages'] = []
    if 'runtime' not in st.session_state:
        if demo or st.sidebar.button('Connect workspace'):
            try:
                with st.spinner('Opening workspace…'):
                    st.session_state['runtime'] = create_runtime(demo=demo)
            except Exception as exc:
                st.error(str(exc))
        if 'runtime' not in st.session_state:
            st.info('Configure .env with your Neo4j connection and model API key, then connect the workspace.')
            st.stop()
    runtime = st.session_state['runtime']
    if demo:
        st.info('Offline demo: extraction and answers are scripted demonstrations. Select Neo4j + Chroma to use the full AI pipeline.')
    with st.sidebar:
        st.subheader('Documents')
        uploads = st.file_uploader('Choose documents', type=['txt', 'md', 'pdf', 'docx', 'csv', 'xlsx'],
                                   accept_multiple_files=True)
        replace = st.checkbox('Replace existing document versions', value=False)
        ingest = st.button('Ingest documents', disabled=not uploads)
        sample = st.button('Load sample document') if demo else False
        if ingest or sample:
            paths = [ROOT / 'data' / 'raw_documents' / 'demo_atlas.txt'] if sample else []
            with tempfile.TemporaryDirectory(prefix='memory-graph-') as folder:
                for upload in uploads if ingest else []:
                    name = Path(upload.name.replace('\\', '/')).name
                    path = Path(folder) / name
                    path.write_bytes(upload.getvalue())
                    paths.append(path)
                for path in paths:
                    with st.spinner(f'Ingesting {path.name}…'):
                        report = runtime.ingestion.ingest_document(path, replace=replace)
                    if report['status'] == 'failed':
                        st.error(f"{path.name}: ingestion failed or incomplete")
                    else:
                        st.success(f"{path.name}: {report['status']}")
                    if report['vector_pending']:
                        st.warning(f"{report['vector_pending']} notes await search indexing.")
                    with st.expander(f'{path.name} report'):
                        st.json(report)
        if st.button('Retry pending indexing'):
            errors = runtime.ingestion.graph_agent.flush_vector_outbox()
            st.error(errors) if errors else st.success('Search index is up to date.')
        graph = runtime.graph_store
        st.caption(f"{len(graph.documents)} documents · {len(graph.notes)} notes · {len(graph.nodes)} entities")
        if st.button('Clear conversation'):
            st.session_state['messages'] = []
            st.rerun()
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            if message['role'] == 'assistant':
                show_response(st, message['response'])
            else:
                st.markdown(message['content'])
    question = st.chat_input('Ask about your documents')
    if question:
        history = [{'role': m['role'], 'content': m['content']} for m in st.session_state['messages']]
        st.session_state['messages'].append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            try:
                with st.spinner('Finding evidence and checking the answer…'):
                    response = runtime.query.answer_question(question, history)
                show_response(st, response)
                st.session_state['messages'].append({'role': 'assistant', 'content': response['answer_text'],
                                                    'response': response})
            except Exception as exc:
                st.error(f'Could not answer: {exc}')


if __name__ == "__main__":
    main()

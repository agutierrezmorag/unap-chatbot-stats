import os

import pandas as pd
import streamlit as st
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.langsmith.tracing
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.langsmith.endpoint
os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith.api_key
os.environ["LANGCHAIN_PROJECT"] = st.secrets.langsmith.project


if __name__ == "__main__":
    st.set_page_config(
        page_title="Langsmith Data",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("📊 UNAP Chatbot Langsmith Runs Data")
    st.markdown("Tracings registrados en la organizacion **unap-chabot** en Langsmith.")
    client = Client()

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        env_type = st.multiselect(
            "Tags",
            ["WebApp Chat", "Test Chat", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview"],
            default=None,
            placeholder="Elija uno o mas tags",
        )

    if env_type:
        if len(env_type) == 1:
            filter_string = f"has(tags, '{env_type[0]}')"
        else:
            filter_string = ", ".join([f"has(tags, '{tag}')" for tag in env_type])
            filter_string = f"and({filter_string})"

        try:
            project_list = client.list_runs(
                project_name=st.secrets.langsmith.project,
                execution_order=1,
                filter=filter_string,
            )
        except Exception as e:
            st.error(e)
            st.stop()
    else:
        project_list = client.list_runs(
            project_name=st.secrets.langsmith.project, execution_order=1
        )

    # Define the attributes of interest
    project_list = [
        {
            "name": str(run.name),
            "run_type": str(run.run_type),
            "tags": run.tags,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
            "extra": run.extra,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "status": run.status,
            "feedback_stats": run.feedback_stats,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "total_tokens": run.total_tokens,
        }
        for run in project_list
    ]

    # Convert project_list to a DataFrame
    project_list_df = pd.DataFrame(project_list)

    # Normalize the 'outputs' column and join it to the original DataFrame
    outputs_df = pd.json_normalize(project_list_df["outputs"].tolist(), sep="_")
    project_list_df = project_list_df.drop(columns="outputs").join(outputs_df)
    # Normalize the 'extra' column and join it to the original DataFrame
    extra_df = pd.json_normalize(project_list_df["extra"].tolist(), sep="_")
    project_list_df = project_list_df.drop(columns="extra").join(extra_df)

    # Create a new column 'total' with the calculated values of 'prompt_tokens' and 'completion_tokens'
    project_list_df["total"] = ((project_list_df["prompt_tokens"] / 1000) * 0.0010) + (
        (project_list_df["completion_tokens"] / 1000) * 0.0020
    )

    # Filter the DataFrame according to the selected status
    with fcol2:
        filter_by_status = st.selectbox("Estado", ["--", "success", "error"])
    if filter_by_status != "--":
        project_list_df = project_list_df[project_list_df["status"] == filter_by_status]

    sessions = project_list_df["metadata_user_session"].unique()
    with fcol3:
        filter_by_session = st.multiselect(
            "Sesion", list(sessions), placeholder="Elija una o mas sesiones"
        )

    if filter_by_session:
        project_list_df = project_list_df[
            project_list_df["metadata_user_session"].isin(filter_by_session)
        ]

    # Display the DataFrame
    st.dataframe(
        project_list_df,
        use_container_width=True,
        column_order=[
            "name",
            "run_type",
            "question",
            "answer",
            "status",
            "error",
            "feedback_stats",
            "start_time",
            "end_time",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "total",
            "tags",
            "metadata_user_session",
            "chat_history",
            "context",
        ],
        column_config={
            "name": st.column_config.Column("Nombre", width="small"),
            "run_type": st.column_config.Column("Tipo", width="small"),
            "question": st.column_config.Column("Pregunta", width="medium"),
            "answer": st.column_config.Column("Respuesta", width="medium"),
            "status": st.column_config.Column("Estado", width="small"),
            "error": st.column_config.Column("Error", width="snmall"),
            "feedback_stats": "Feedback",
            "start_time": st.column_config.DatetimeColumn(
                "Inicio", format="D MMM YYYY, HH:mm:ss", width="small"
            ),
            "end_time": st.column_config.DatetimeColumn(
                "Fin", format="D MMM YYYY, HH:mm:ss", width="small"
            ),
            "prompt_tokens": "Tokens de pregunta",
            "completion_tokens": "Tokens de respuesta",
            "total_tokens": "Tokens totales",
            "total": st.column_config.NumberColumn(
                "Costo (USD)", format="%.4f", width="small"
            ),
            "tags": st.column_config.Column("Tags", width="small"),
            "metadata_user_session": st.column_config.Column("Sesion", width="small"),
            "chat_history": st.column_config.Column("Historial", width="small"),
            "context": st.column_config.Column("Contexto", width="small"),
        },
    )

    # Calculate the total values
    total_runs = len(project_list_df)
    total_tokens_sum = project_list_df["total_tokens"].sum()
    total_sum = project_list_df["total"].sum()

    total_tokens_avg = project_list_df["total_tokens"].mean()
    total_cost_avg = project_list_df["total"].mean()

    success_percentage = (
        project_list_df["status"].value_counts(normalize=True)["success"] * 100
    )

    status_counts = project_list_df["status"].value_counts()
    successful_runs = status_counts["success"]
    total_runs = status_counts.sum()

    # Display the total values
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total de ejecuciones:", total_runs)
    with col2:
        st.metric(
            "Ejecuciones Exitosas:",
            f"{success_percentage:.2f}%",
            help=f"{success_percentage}",
        )
        st.markdown(f"**{successful_runs}** de {total_runs} runs fueron exitosas")
    with col3:
        st.metric(
            "Tokens Promedio:", f"{total_tokens_avg:.2f}", help=f"{total_tokens_avg}"
        )
    with col4:
        st.metric("Tokens Totales:", total_tokens_sum)
    with col5:
        st.metric(
            "Costo Promedio (USD):", f"{total_cost_avg:.4f}", help=f"{total_cost_avg}"
        )
    with col6:
        st.metric("Costo Total (USD):", f"{total_sum:.4f}", help=f"{total_sum}")

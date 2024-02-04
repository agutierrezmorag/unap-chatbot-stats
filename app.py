import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

column_order = [
    "chat_type",
    "question",
    "answer",
    "sources",
    "time_to_answer",
    "tokens_total_tokens",
    "tokens_total_cost_usd",
    "user_feedback_score",
    "user_feedback_text",
    "submission_time",
    "evaluate",
]

column_config = {
    "chat_type": st.column_config.Column(
        "Tipo de chat",
        width="small",
    ),
    "question": st.column_config.Column(
        "Pregunta",
        width="small",
    ),
    "answer": "Respuesta",
    "sources": "Fuentes",
    "time_to_answer": st.column_config.NumberColumn(
        "Tiempo de respuesta (s)",
        format="%.2f",
        width="small",
        help="Este tiempo considera desde que el usuario realiza la pregunta hasta que la respuesta es mostrada en pantalla.",
    ),
    "tokens_total_tokens": st.column_config.Column(
        "Total de tokens",
        width="small",
    ),
    "tokens_total_cost_usd": st.column_config.NumberColumn(
        "Costo (USD)",
        format="$%.3f",
        width="small",
    ),
    "user_feedback_score": st.column_config.Column(
        "Puntaje de usuario",
        width="small",
    ),
    "user_feedback_text": st.column_config.Column(
        "Feedback",
        help="Comentario opcional proporcionado por el usuario.",
    ),
    "submission_time": st.column_config.DatetimeColumn(
        "Fecha",
        format="D MMM YYYY, h:mm a",
        timezone="America/Santiago",
        help="Fecha y hora en que se proceso la pregunta.",
    ),
    "evaluate": st.column_config.CheckboxColumn(
        "Evaluar",
        help="Selecciona para evaluar la calidad de la respuesta, comparandola con los documentos consultados para su generacion.",
        default=False,
        width="small",
    ),
}


@st.cache_resource
def db_connection():
    key_dict = json.loads(st.secrets.firestore.textkey)
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


@st.cache_data(ttl=60 * 60 * 3)
def load_messages():
    with open("messages.json", "r", encoding="utf-8") as f:
        messages = json.load(f)
    return messages


def get_avg_time_to_answer(df):
    return df["time_to_answer"].mean()


def get_message_count(df):
    return len(df)


def get_total_cost(df):
    return df["tokens_total_cost_usd"].sum()


@st.cache_resource
def get_comparison_model():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model


def perform_comparison(df):
    model = get_comparison_model()

    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        # Initialize lists to store the data
        answers = []
        compared_contexts = []
        similarity_scores = []

        # Initialize lists to store the data for the combined context comparison
        combined_answers = []
        combined_contexts = []
        combined_scores = []

        # Get the embedding for the 'answer'
        answer_embedding = model.encode([row["answer"]], convert_to_tensor=True)

        # Extract the 'context' values from the 'sources'
        if row["sources"] is not None:
            contexts = [d["context"] for d in row["sources"]]
            context_embeddings = model.encode(contexts, convert_to_tensor=True)

            # Calculate the cosine similarity for each 'context'
            scores = cosine_similarity(answer_embedding, context_embeddings)

            # Flatten the scores and add the data to the lists
            scores_flat = scores.flatten().tolist()
            answers.extend([row["answer"]] * len(contexts))
            compared_contexts.extend(contexts)
            similarity_scores.extend(scores_flat)

            # Combine all the 'context' values into one string and get its embedding
            combined_context = " ".join(contexts)
            combined_context_embedding = model.encode(
                [combined_context], convert_to_tensor=True
            )

            # Calculate the cosine similarity for the combined 'context'
            combined_score = cosine_similarity(
                answer_embedding, combined_context_embedding
            )

            # Flatten the score and add the data to the lists
            combined_score_flat = combined_score.flatten()[0]
            combined_answers.append(row["answer"])
            combined_contexts.append(combined_context)
            combined_scores.append(combined_score_flat)

        # Create a DataFrame with the compared values and their scores
        compared_values_df = pd.DataFrame(
            {
                "Respuesta": answers,
                "Contexto": compared_contexts,
                "Similarity Score": similarity_scores,
            }
        )

        # Create a DataFrame with the compared values and their scores for the combined context comparison
        combined_values_df = pd.DataFrame(
            {
                "Respuesta": combined_answers,
                "Contexto (concatenado)": combined_contexts,
                "Similarity Score": combined_scores,
            }
        )

        # Display the DataFrames
        st.markdown(f"## Evaluacion de respuesta: {row['question']}")
        st.dataframe(compared_values_df, use_container_width=True, hide_index=True)
        st.dataframe(combined_values_df, use_container_width=True, hide_index=True)
        st.divider()


def main():
    st.set_page_config(page_title="Chatbot data", page_icon="📈", layout="wide")
    st.title("📊 UNAP Chatbot Data")
    st.markdown("Visualizacion de todos los datos registrados.")

    messages = load_messages()
    df = pd.json_normalize(messages, sep="_")
    df["submission_time"] = pd.to_datetime(df["submission_time"])
    df["message_id"] = df["message_id"].astype(str)
    checkbox_ids = [False for _ in range(len(df))]
    df["evaluate"] = checkbox_ids

    col1, col2, col3 = st.columns(3)

    with col1:
        chat_types = df["chat_type"].unique().tolist()
        selected_chat_types = st.multiselect(
            "Tipo de chat",
            options=chat_types,
            placeholder="Seleccione uno o más tipos de chat",
        )

    with col2:
        user_scores = df["user_feedback_score"].unique().tolist()
        user_scores.insert(0, "--")
        selected_user_score = st.selectbox("Puntaje de usuario", options=user_scores)

    with col3:
        scol1, scol2 = st.columns(2)
        with scol1:
            start_date = st.date_input(
                "Fecha de inicio",
                None,
                format="DD/MM/YYYY",
                min_value=df["submission_time"].min().date(),
                max_value=df["submission_time"].max().date(),
            )
        with scol2:
            end_date = st.date_input(
                "Fecha de término",
                None,
                format="DD/MM/YYYY",
                min_value=df["submission_time"].min().date(),
                max_value=df["submission_time"].max().date(),
            )

    col4, col5 = st.columns(2)
    with col4:
        question_query = st.text_input("Buscar por pregunta...")

    with col5:
        answer_query = st.text_input("Buscar por respuesta...")

    if not selected_chat_types:
        filtered_messages = df
    else:
        filtered_messages = df[df["chat_type"].isin(selected_chat_types)]

    if selected_user_score != "--":
        filtered_messages = filtered_messages[
            filtered_messages["user_feedback_score"] == selected_user_score
        ]

    message_display_count = st.slider(
        "Numero de mensajes a considerar",
        help="Se consideraran los ultimos mensajes, segun la cantidad seleccionada.",
        min_value=1,
        max_value=len(filtered_messages),
        value=len(filtered_messages),
    )

    if question_query:
        filtered_messages = filtered_messages[
            filtered_messages["question"]
            .fillna("")
            .str.contains(question_query, case=False)
        ]

    if answer_query:
        filtered_messages = filtered_messages[
            filtered_messages["question"]
            .fillna("")
            .str.contains(answer_query, case=False)
        ]

    if start_date and end_date:
        if start_date > end_date:
            st.error("Error: La fecha de inicio debe ser previo a la fecha de término.")
        else:
            filtered_messages = df[
                (df["submission_time"].dt.date >= start_date)
                & (df["submission_time"].dt.date <= end_date)
            ]

    filtered_messages["sources"] = filtered_messages["sources"].apply(
        lambda x: x if x is not None else None
    )

    st.data_editor(
        filtered_messages.tail(message_display_count),
        key="data_editor",
        use_container_width=True,
        column_order=column_order,
        column_config=column_config,
    )

    changed_rows = []
    for row in st.session_state.data_editor["edited_rows"]:
        changed_rows.append(row)
    st.write(filtered_messages.loc[changed_rows])
    if st.button("Evaluar"):
        perform_comparison(filtered_messages.loc[changed_rows])

    avg_time_to_answer_all = df["time_to_answer"].mean()
    avg_time_to_answer_selected = filtered_messages["time_to_answer"].mean()

    delta_time_to_answer = avg_time_to_answer_selected - avg_time_to_answer_all

    total_message_count = len(df)
    selected_message_count = len(filtered_messages)
    delta_message_count = selected_message_count - total_message_count

    total_cost_all = df["tokens_total_cost_usd"].sum()
    total_cost_selected = filtered_messages["tokens_total_cost_usd"].sum()
    delta_cost = total_cost_selected - total_cost_all

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Tiempo promedio de respuesta",
            round(avg_time_to_answer_selected, 2),
            delta=delta_time_to_answer,
            delta_color="inverse",
            help="Tiempo promedio de respuesta de lo filtrado comparado con todos los mensajes.",
        )
    with col2:
        st.metric(
            "Mensajes",
            selected_message_count,
            delta=delta_message_count,
            delta_color="off",
            help="Tiempo promedio de respuesta de lo filtrado comparado con todos los mensajes.",
        )
    with col3:
        st.metric(
            "Costo total (USD)",
            5.327222 + round(total_cost_selected, 2),
            delta=delta_cost,
            help="Costo total de lo filtrado comparado con todos los mensajes.",
        )

    st.markdown("# 💸 Comparación de costos")
    st.markdown("Comparacion de metricas entre dos tipos de chat.")

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        chat_type1 = st.selectbox("Comparar...", options=df["chat_type"].unique())
    with comp_col2:
        chat_type2 = st.selectbox("Con...", options=df["chat_type"].unique())
    with comp_col3:
        num_messages = st.number_input(
            "Numero de mensajes", min_value=1, value=10, step=1
        )

    # Filter the DataFrame based on the selected chat types and number of messages
    df1 = df[df["chat_type"] == chat_type1].tail(num_messages)
    df2 = df[df["chat_type"] == chat_type2].tail(num_messages)

    # Calculate metrics for each chat type
    avg_time_to_answer1 = get_avg_time_to_answer(df1)
    avg_time_to_answer2 = get_avg_time_to_answer(df2)

    message_count1 = get_message_count(df1)
    message_count2 = get_message_count(df2)

    total_cost1 = get_total_cost(df1)
    total_cost2 = get_total_cost(df2)

    # Display metrics side by side
    metrics = [
        "Tiempo promedio de respuesta",
        "Mensajes",
        "Costo total (USD)",
    ]
    values1 = [avg_time_to_answer1, message_count1, total_cost1]
    values2 = [avg_time_to_answer2, message_count2, total_cost2]

    cols = st.columns(len(metrics))

    for metric, value1, value2, col in zip(metrics, values1, values2, cols):
        plt.figure(figsize=(6, 4))
        bars = plt.bar(
            [chat_type1, chat_type2], [value1, value2], color=["#0069B1", "#11B334"]
        )
        plt.title(metric)
        plt.ylabel(metric)

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va="bottom"
            )

        col.pyplot(plt)

    dfcol1, dfcol2 = st.columns(2)

    with dfcol1:
        st.dataframe(
            df1,
            use_container_width=True,
            column_order=column_order,
            column_config=column_config,
        )

    with dfcol2:
        st.dataframe(
            df2,
            use_container_width=True,
            column_order=column_order,
            column_config=column_config,
        )


if __name__ == "__main__":
    main()

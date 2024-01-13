import json

import pandas as pd
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account


@st.cache_resource
def db_connection():
    key_dict = json.loads(st.secrets.firestore.textkey)
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


@st.cache_data(ttl=60 * 60 * 3)
def load_messages():
    with open("messages.json", "r") as f:
        messages = json.load(f)
    return messages


def get_chat_count(df):
    return len(df)


def get_avg_time_to_answer(df):
    return df["time_to_answer"].mean()


def get_message_count(df):
    return len(df)


def get_total_cost(df):
    return df["tokens_total_cost_usd"].sum()


def main():
    st.set_page_config(page_title="Chatbot data", page_icon="📈", layout="wide")
    st.title("📊 UNAP Chatbot Data")

    messages = load_messages()
    df = pd.json_normalize(messages, sep="_")
    df["submission_time"] = pd.to_datetime(df["submission_time"])

    col1, col2, col3 = st.columns(3)

    with col1:
        chat_types = df["chat_type"].unique().tolist()
        chat_types.insert(0, "--")
        selected_chat_type = st.selectbox("Tipo de chat", options=chat_types)

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
                min_value=df["submission_time"].min().date(),
                max_value=df["submission_time"].max().date(),
            )
        with scol2:
            end_date = st.date_input(
                "Fecha de término",
                None,
                min_value=df["submission_time"].min().date(),
                max_value=df["submission_time"].max().date(),
            )

    col4, col5 = st.columns(2)
    with col4:
        question_query = st.text_input("Buscar por pregunta...")

    with col5:
        answer_query = st.text_input("Buscar por respuesta...")

    if selected_chat_type == "--":
        filtered_messages = df
    else:
        filtered_messages = df[df["chat_type"] == selected_chat_type]

    if selected_user_score == "--":
        filtered_messages = filtered_messages
    else:
        filtered_messages = filtered_messages[
            filtered_messages["user_feedback_score"] == selected_user_score
        ]

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
            st.error("Error: La fecha de inicio debe ser menor a la fecha de término.")
        elif start_date and end_date:
            filtered_messages = df[
                (df["submission_time"].dt.date >= start_date)
                & (df["submission_time"].dt.date <= end_date)
            ]

    avg_time_to_answer_all = df["time_to_answer"].mean()
    avg_time_to_answer_selected = filtered_messages["time_to_answer"].mean()

    delta_time_to_answer = avg_time_to_answer_selected - avg_time_to_answer_all

    total_message_count = len(df)
    selected_message_count = len(filtered_messages)
    delta_message_count = selected_message_count - total_message_count

    total_cost_all = df["tokens_total_cost_usd"].sum()
    total_cost_selected = filtered_messages["tokens_total_cost_usd"].sum()
    delta_cost = total_cost_selected - total_cost_all

    st.write(filtered_messages)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chats", len(df))
    with col2:
        st.metric(
            "Tiempo promedio de respuesta",
            avg_time_to_answer_selected,
            delta=delta_time_to_answer,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Mensajes",
            selected_message_count,
            delta=delta_message_count,
        )
    with col4:
        st.metric(
            "Costo total en USD",
            total_cost_selected,
            delta=delta_cost,
        )

    st.markdown("# 💸 Comparación de costos")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        chat_type1 = st.selectbox("Comparar...", options=df["chat_type"].unique())
    with comp_col2:
        chat_type2 = st.selectbox("Con...", options=df["chat_type"].unique())

    # Calculate metrics for each chat type
    chat_count1 = get_chat_count(df[df["chat_type"] == chat_type1])
    chat_count2 = get_chat_count(df[df["chat_type"] == chat_type2])

    avg_time_to_answer1 = get_avg_time_to_answer(df[df["chat_type"] == chat_type1])
    avg_time_to_answer2 = get_avg_time_to_answer(df[df["chat_type"] == chat_type2])

    message_count1 = get_message_count(df[df["chat_type"] == chat_type1])
    message_count2 = get_message_count(df[df["chat_type"] == chat_type2])

    total_cost1 = get_total_cost(df[df["chat_type"] == chat_type1])
    total_cost2 = get_total_cost(df[df["chat_type"] == chat_type2])

    # Display metrics side by side
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chats", chat_count1, delta=chat_count2 - chat_count1)
    with col2:
        st.metric(
            "Tiempo promedio de respuesta",
            avg_time_to_answer1,
            delta=avg_time_to_answer2 - avg_time_to_answer1,
        )
    with col3:
        st.metric("Mensajes", message_count1, delta=message_count2 - message_count1)
    with col4:
        st.metric("Costo total en USD", total_cost1, delta=total_cost2 - total_cost1)


if __name__ == "__main__":
    main()

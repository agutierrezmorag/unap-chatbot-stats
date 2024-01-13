import datetime
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


def get_messages():
    db = db_connection()
    chats_ref = db.collection("chats")
    chats = chats_ref.stream()

    sorted_chats = sorted(chats, key=lambda chat: int(chat.id))

    all_messages = []
    for chat in sorted_chats:
        messages_ref = db.collection("chats").document(chat.id).collection("messages")
        messages = messages_ref.stream()
        for message in messages:
            message_dict = message.to_dict()
            for key, value in message_dict.items():
                if isinstance(value, datetime.datetime):
                    message_dict[key] = value.isoformat()
            all_messages.append(message_dict)

    with open("messages.json", "w") as f:
        json.dump(all_messages, f)

    return all_messages


@st.cache_data(ttl=60 * 60 * 3)
def load_messages():
    with open("messages.json", "r") as f:
        messages = json.load(f)
    return messages


def get_chat_count():
    db = db_connection()
    chats_ref = db.collection("chats")
    chats = chats_ref.stream()
    return len(list(chats))


def main():
    st.set_page_config(page_title="Chatbot data", page_icon="📈", layout="wide")
    st.title("📊 UNAP Chatbot Data")
    messages = load_messages()
    df = pd.json_normalize(messages, sep="_")

    chat_types = df["chat_type"].unique().tolist()
    chat_types.insert(0, "Todo")
    selected_chat_type = st.selectbox("Tipo de chat", options=chat_types)

    if selected_chat_type == "Todo":
        filtered_messages = df
    else:
        filtered_messages = df[df["chat_type"] == selected_chat_type]

    avg_time_to_answer_all = df["time_to_answer"].mean()
    avg_time_to_answer_selected = filtered_messages["time_to_answer"].mean()

    delta_time_to_answer = avg_time_to_answer_selected - avg_time_to_answer_all

    total_message_count = len(df)
    selected_message_count = len(filtered_messages)
    delta_message_count = selected_message_count - total_message_count

    total_cost_all = df["tokens_total_cost_usd"].sum()
    total_cost_selected = filtered_messages["tokens_total_cost_usd"].sum()
    delta_cost = total_cost_selected - total_cost_all

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chats", get_chat_count())
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

    st.write(filtered_messages)


if __name__ == "__main__":
    main()

import datetime
import json
import os

from google.cloud import firestore
from google.oauth2 import service_account


def db_connection():
    key_dict = json.loads(os.getenv("FIRESTORE_TEXTKEY"))
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


def get_messages():
    db = db_connection()
    chats_ref = db.collection("chats")
    chats = chats_ref.stream()

    sorted_chats = sorted(chats, key=lambda chat: int(chat.id))
    total_chats = len(sorted_chats)

    all_messages = []
    try:
        for i, chat in enumerate(sorted_chats, start=1):
            messages_ref = (
                db.collection("chats").document(chat.id).collection("messages")
            )
            messages = messages_ref.stream()
            for message in messages:
                message_dict = message.to_dict()
                for key, value in message_dict.items():
                    if isinstance(value, datetime.datetime):
                        message_dict[key] = value.isoformat()

                # Fetch the documents from the 'sources' sub-collection
                sources_ref = messages_ref.document(message.id).collection("sources")
                sources = sources_ref.stream()
                sources_list = [source.to_dict() for source in sources]
                message_dict["sources"] = sources_list

                all_messages.append(message_dict)
            print(f"{i}/{total_chats} done.")
    except Exception as e:
        print(e)

    with open("messages.json", "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False)

    return all_messages


if __name__ == "__main__":
    get_messages()

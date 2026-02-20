import json
import re
from datetime import datetime, timezone, timedelta

import streamlit as st
from openai import OpenAI

APP_TITLE = "Streamlit LLM"
DEFAULT_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o-mini",
]

JST = timezone(timedelta(hours=9))


def _now_jst_compact() -> str:
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "chat_log"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120]


def _looks_like_api_key(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 20:
        return False
    if not any(ch.isalnum() for ch in s):
        return False
    return True


def init_state() -> None:
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODELS[0]
    if "mode" not in st.session_state:
        st.session_state.mode = "会話を受け継ぐ"

    if "single_answer" not in st.session_state:
        st.session_state.single_answer = ""

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "active_session" not in st.session_state:
        st.session_state.active_session = "default"
    if st.session_state.active_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.active_session] = []

    if "global_context" not in st.session_state:
        st.session_state.global_context = ""
    if "session_contexts" not in st.session_state:
        st.session_state.session_contexts = {}
    if st.session_state.active_session not in st.session_state.session_contexts:
        st.session_state.session_contexts[st.session_state.active_session] = ""

    st.session_state.chat_messages = st.session_state.chat_sessions[st.session_state.active_session]

    if "session_context_editor" not in st.session_state:
        st.session_state.session_context_editor = st.session_state.session_contexts.get(
            st.session_state.active_session, ""
        )


def save_current_session() -> None:
    st.session_state.chat_sessions[st.session_state.active_session] = st.session_state.chat_messages


def build_client() -> OpenAI | None:
    ui_key = st.session_state.api_key.strip()
    if _looks_like_api_key(ui_key):
        return OpenAI(api_key=ui_key)
    return None


def export_active_session_json() -> str:
    save_current_session()
    data = {
        "exported_at_jst": datetime.now(JST).isoformat(),
        "model": st.session_state.get("model", ""),
        "active_session": st.session_state.active_session,
        "chat_sessions": st.session_state.chat_sessions,
        "global_context": st.session_state.get("global_context", ""),
        "session_contexts": st.session_state.get("session_contexts", {}),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def import_log_json(uploaded_file, merge_default: bool = True, target_session: str | None = None) -> None:
    raw = uploaded_file.read().decode("utf-8")
    data = json.loads(raw)

    incoming_sessions = data.get("chat_sessions")
    incoming_active = data.get("active_session")

    if target_session is None:
        target_session = incoming_active or st.session_state.active_session

    if not isinstance(target_session, str) or not target_session.strip():
        target_session = st.session_state.active_session
    target_session = target_session.strip()

    if incoming_sessions is None:
        incoming_msgs = data.get("chat_messages", [])
        if not isinstance(incoming_msgs, list):
            raise ValueError("chat_messages が list ではありません。")
        incoming_sessions = {target_session: incoming_msgs}

    if not isinstance(incoming_sessions, dict):
        raise ValueError("chat_sessions が dict ではありません。")

    if merge_default:
        for sname, msgs in incoming_sessions.items():
            if not isinstance(sname, str) or not isinstance(msgs, list):
                continue
            if sname not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[sname] = []
            st.session_state.chat_sessions[sname].extend(msgs)
    else:
        if target_session in incoming_sessions and isinstance(incoming_sessions[target_session], list):
            st.session_state.chat_sessions[target_session] = incoming_sessions[target_session]
        else:
            raise ValueError("指定セッションが見つかりません。")

    incoming_global = data.get("global_context")
    if isinstance(incoming_global, str) and incoming_global.strip():
        if merge_default and st.session_state.get("global_context", "").strip():
            st.session_state.global_context = st.session_state.global_context.rstrip() + "\n\n" + incoming_global.strip()
        else:
            st.session_state.global_context = incoming_global

    incoming_session_ctx = data.get("session_contexts")
    if isinstance(incoming_session_ctx, dict):
        for sname, ctx in incoming_session_ctx.items():
            if not isinstance(sname, str) or not isinstance(ctx, str):
                continue
            if merge_default and st.session_state.session_contexts.get(sname, "").strip():
                st.session_state.session_contexts[sname] = (
                    st.session_state.session_contexts[sname].rstrip() + "\n\n" + ctx.strip()
                )
            else:
                st.session_state.session_contexts[sname] = ctx

    if target_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[target_session] = []
    if target_session not in st.session_state.session_contexts:
        st.session_state.session_contexts[target_session] = ""

    st.session_state.active_session = target_session
    st.session_state.chat_messages = st.session_state.chat_sessions[target_session]
    st.session_state.session_context_editor = st.session_state.session_contexts.get(target_session, "")

    st.session_state.chat_messages = [m for m in st.session_state.chat_messages if m.get("role") != "system"]
    save_current_session()


def build_messages_for_api() -> list[dict]:
    current = st.session_state.active_session
    g = st.session_state.global_context.strip()
    s = st.session_state.session_contexts.get(current, "").strip()

    sys_parts = []
    if g:
        sys_parts.append("全体共通の前提知識:\n" + g)
    if s:
        sys_parts.append("このセッションの前提知識:\n" + s)

    system_text = "\n\n".join(sys_parts).strip()

    msgs: list[dict] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.extend(st.session_state.chat_messages)
    return msgs


def render_session_manager_sidebar() -> None:
    with st.sidebar:
        st.header("セッション管理")

        existing = list(st.session_state.chat_sessions.keys())
        if st.session_state.active_session not in existing:
            existing = [st.session_state.active_session] + existing

        selected = st.selectbox(
            "現在のセッション",
            existing,
            index=existing.index(st.session_state.active_session),
            key="session_selectbox",
        )
        if selected != st.session_state.active_session:
            save_current_session()
            st.session_state.active_session = selected
            st.session_state.chat_messages = st.session_state.chat_sessions[selected]
            if selected not in st.session_state.session_contexts:
                st.session_state.session_contexts[selected] = ""
            st.session_state.session_context_editor = st.session_state.session_contexts.get(selected, "")

        new_name = st.text_input("新しいセッション名", value="", key="new_session_name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("新規作成", use_container_width=True, key="create_session_btn"):
                name = new_name.strip() or f"session_{_now_jst_compact()}"
                if name in st.session_state.chat_sessions:
                    st.warning("同名セッションが既にあります。")
                else:
                    save_current_session()
                    st.session_state.chat_sessions[name] = []
                    st.session_state.session_contexts[name] = ""
                    st.session_state.active_session = name
                    st.session_state.chat_messages = st.session_state.chat_sessions[name]
                    st.session_state.session_context_editor = ""
        with col2:
            if st.button("このセッションを空にする", use_container_width=True, key="clear_session_btn"):
                st.session_state.chat_sessions[st.session_state.active_session] = []
                st.session_state.chat_messages = st.session_state.chat_sessions[st.session_state.active_session]
                save_current_session()

        st.divider()
        st.header("前提知識")

        st.text_area(
            "全体共通の前提知識",
            key="global_context",
            height=140,
            placeholder="例: 呼び方, 出力方針, 注意事項など",
        )

        current = st.session_state.active_session
        if current not in st.session_state.session_contexts:
            st.session_state.session_contexts[current] = ""

        st.text_area(
            "このセッションの前提知識",
            key="session_context_editor",
            height=140,
            placeholder="例: プロジェクトの目的, 成果物の形式, 禁止事項など",
        )

        if st.button("前提知識をこのセッションに反映", use_container_width=True, key="apply_session_context_btn"):
            st.session_state.session_contexts[current] = st.session_state.session_context_editor
            st.success("反映しました。")

        st.divider()
        st.header("ログ入出力")

        default_fname = f"chat_{_safe_filename(st.session_state.active_session)}_{_now_jst_compact()}.json"
        fname_in = st.text_input("保存ファイル名", value=default_fname, key="export_filename")
        fname = _safe_filename(fname_in)
        if not fname.lower().endswith(".json"):
            fname = f"{fname}.json"

        json_text = export_active_session_json()
        st.download_button(
            label="ログをJSONでダウンロード",
            data=json_text,
            file_name=fname,
            mime="application/json",
            use_container_width=True,
            key="download_log_btn",
        )

        st.caption("読み込みは既存に追加が基本です。")
        merge_flag = st.checkbox("既存に追加する", value=True, key="merge_import_chk")

        up = st.file_uploader("ログを読み込む", type=["json"], key="log_uploader")
        import_to = st.text_input(
            "読み込み先セッション名",
            value=st.session_state.active_session,
            key="import_target_session",
        )

        if up is not None:
            if st.button("このファイルを読み込む", use_container_width=True, key="do_import_btn"):
                try:
                    import_log_json(up, merge_default=merge_flag, target_session=import_to.strip() or None)
                    st.success("読み込みました。")
                except Exception as e:
                    st.error(f"読み込みに失敗しました: {e}")


def render_setup() -> None:
    st.title(APP_TITLE)

    st.subheader("安全性とプライバシー")
    st.write("このアプリは OpenAI API を呼び出すだけのフロントエンドです。")
    st.write("APIキーはログファイルやダウンロードJSONに保存しません。")
    st.write("APIキーは画面に表示しません。")
    st.write("APIキーはこのブラウザセッション内でのみ保持されます。")
    st.write("開発者の深澤は、ユーザーのAPIキーや入力内容を回収しません。")
    st.write("このアプリは OpenAI API に対して、入力された文章と前提知識と会話履歴を送信します。")

    st.subheader("APIキー")
    st.text_input(
        "OpenAI APIキー",
        type="password",
        key="api_key",
        placeholder="sk-...",
        help="必ずユーザー自身のAPIキーを入力してください。入力内容はセッション内だけで保持されます。",
    )

    st.subheader("モデル選択の目安")
    st.write("高性能モデルは精度が高い傾向があります。")
    st.write("一方で消費が大きく、利用上限や課金枠を早く使い切りやすいです。")
    st.write("軽量モデルは消費を抑えやすいです。")
    st.write("一方で精度が足りない場合があります。")

    st.selectbox(
        "モデル",
        DEFAULT_MODELS,
        key="model",
        help="使えるモデルは契約や権限で変わります。",
    )

    st.radio(
        "モード",
        ["会話を受け継ぐ", "単一で質問に答える"],
        key="mode",
        horizontal=True,
    )

    cols = st.columns(3)
    with cols[0]:
        if st.button("会話をリセット", use_container_width=True, key="reset_chat_btn"):
            st.session_state.chat_messages = []
            save_current_session()
    with cols[1]:
        if st.button("単発の回答をクリア", use_container_width=True, key="clear_single_btn"):
            st.session_state.single_answer = ""
    with cols[2]:
        if st.button("APIキーを消去", use_container_width=True, key="clear_key_btn"):
            st.session_state.api_key = ""


def run_chat_mode(client: OpenAI) -> None:
    st.subheader("会話モード")

    st.session_state.chat_messages = [m for m in st.session_state.chat_messages if m.get("role") != "system"]
    save_current_session()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.write(msg.get("content", ""))

    user_text = st.chat_input("メッセージを入力")
    if not user_text:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    save_current_session()
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("生成中"):
            try:
                completion = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=build_messages_for_api(),
                )
                answer = completion.choices[0].message.content or ""
            except Exception as e:
                answer = f"エラー: {e}"

        st.write(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        save_current_session()


def run_single_mode(client: OpenAI) -> None:
    st.subheader("単発モード")

    prompt = st.text_area(
        "質問",
        height=140,
        placeholder="ここに質問を入力",
        key="single_prompt",
    )

    cols = st.columns(2)
    with cols[0]:
        do_run = st.button("実行", type="primary", use_container_width=True, key="single_run_btn")
    with cols[1]:
        if st.button("クリア", use_container_width=True, key="single_clear_btn"):
            st.session_state.single_answer = ""

    if do_run:
        if not prompt.strip():
            st.warning("質問が空です。")
            return

        with st.spinner("生成中"):
            try:
                resp = client.responses.create(
                    model=st.session_state.model,
                    input=prompt,
                )
                st.session_state.single_answer = resp.output_text
            except Exception as e:
                st.session_state.single_answer = f"エラー: {e}"

    if st.session_state.single_answer:
        st.markdown("回答")
        st.write(st.session_state.single_answer)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    init_state()

    render_session_manager_sidebar()

    with st.sidebar:
        st.header("設定")
        st.write("左の設定を埋めると実行できます。")

    render_setup()

    client = build_client()
    if client is None:
        st.info("APIキーを入力してください。")
        return

    st.divider()

    if st.session_state.mode == "会話を受け継ぐ":
        run_chat_mode(client)
    else:
        run_single_mode(client)


if __name__ == "__main__":
    main()

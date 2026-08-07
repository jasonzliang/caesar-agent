import os
import sys
import sqlite3
import argparse
import random
import pandas as pd
# pip install pysqlite3

# --- 0. Detect Run Mode ---
# When run with `python caesar_human_eval_app.py`, export CSV and exit.
# When run with `streamlit run caesar_human_eval_app.py`, launch the web app.
def _running_under_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except ImportError:
        return False

def export_to_csv(eval_dir, most_recent=False):
    db_file = os.path.join(eval_dir, "results_ab.db")
    export_csv_name = os.path.join(eval_dir, "evaluation_winners.csv")

    if not os.path.exists(db_file):
        print(f"Error: Could not find '{db_file}'.")
        print(f"   Checked directory: {os.path.abspath(eval_dir)}")
        return

    if not most_recent:
        print("Mode: Including ALL evaluation records.")
        query = "SELECT query_file, user_name, winning_file FROM comparative_evals"
    else:
        print("Mode: Filtering for MOST RECENT evaluation per user/query.")
        query = """
            SELECT query_file, user_name, winning_file
            FROM comparative_evals
            WHERE id IN (
                SELECT MAX(id) FROM comparative_evals
                GROUP BY user_name, query_file
            )
        """

    print(f"Connecting to database at {db_file}...")
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("The database is empty. No evaluations to export.")
        return

    df.to_csv(export_csv_name, index=False)
    print(f"Exported {len(df)} records to {export_csv_name}")

    print("\nWin Counts by Specific File:")
    print(df['winning_file'].value_counts().to_string())

    df['model_name'] = df['winning_file'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    print("\nOverall Win Counts by Model:")
    print(df['model_name'].value_counts().to_string())
    print()


def main():
    default_dir = os.environ.get('CAESAR_HUMAN_EVAL_DIR', '.')
    parser = argparse.ArgumentParser(description="Export evaluation results from SQLite to CSV.")
    parser.add_argument("--eval-dir", "-d", type=str, default=default_dir,
                        help=f"Directory containing the DB and where to save CSV (default: {default_dir})")
    parser.add_argument("--most-recent", "-r", action="store_true",
                        help="Include only the most recent evaluations from database.")
    args = parser.parse_args()
    export_to_csv(eval_dir=args.eval_dir, most_recent=args.most_recent)


if not _running_under_streamlit():
    main()
    sys.exit(0)

# --- Streamlit Web App below ---
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(page_title="Model Evaluation (A/B Test)", layout="wide")

# --- 2. Configuration & Validation ---
def _required_subdirs(base):
    return [os.path.join(base, d) for d in ("query", "source_a", "source_b")]

def _is_valid_eval_dir(base):
    return bool(base) and os.path.isdir(base) and all(os.path.isdir(d) for d in _required_subdirs(base))

# Resolve eval directory: env var first, then session state from prior prompt.
if "eval_dir" not in st.session_state:
    st.session_state.eval_dir = os.environ.get('CAESAR_HUMAN_EVAL_DIR', '')

# If no valid path yet, prompt the user.
if not _is_valid_eval_dir(st.session_state.eval_dir):
    st.title("Caesar Human Eval — Setup")
    st.markdown(
        "`CAESAR_HUMAN_EVAL_DIR` is not set, or the directory is missing the "
        "required subdirectories `query/`, `source_a/`, and `source_b/`. "
        "Enter an absolute path to a valid eval directory below."
    )
    new_path = st.text_input(
        "Eval directory path",
        value=st.session_state.eval_dir,
        placeholder="/absolute/path/to/eval_dir",
    )
    if st.button("Continue"):
        new_path = os.path.expanduser(new_path.strip())
        if not new_path:
            st.error("Please enter a path.")
        elif not os.path.isdir(new_path):
            st.error(f"Directory does not exist: {new_path}")
        else:
            missing = [os.path.basename(d) for d in _required_subdirs(new_path) if not os.path.isdir(d)]
            if missing:
                st.error(f"Missing subdirectories under {new_path}: {', '.join(missing)}")
            else:
                st.session_state.eval_dir = new_path
                st.rerun()
    st.stop()

EVAL_DIR = st.session_state.eval_dir
QUERY_DIR = os.path.join(EVAL_DIR, "query")
SOURCE_A_DIR = os.path.join(EVAL_DIR, "source_a")
SOURCE_B_DIR = os.path.join(EVAL_DIR, "source_b")
DB_FILE = os.path.join(EVAL_DIR, "results_ab.db")

# --- 3. Initialize Session State ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "current_pair" not in st.session_state:
    st.session_state.current_pair = 0
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "eval"

# --- 4. Database Functions ---
def init_db():
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comparative_evals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_file TEXT,
                output_1_file TEXT,
                output_2_file TEXT,
                user_name TEXT,
                winner_selection TEXT,
                winning_file TEXT
            )
        ''')
        conn.commit()

def save_result(query_name, out1_file, out2_file, user_name, selection, winning_file):
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO comparative_evals
            (query_file, output_1_file, output_2_file, user_name, winner_selection, winning_file)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_name, out1_file, out2_file, user_name, selection, winning_file))
        conn.commit()

def reset_user_ratings(user_name):
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM comparative_evals WHERE user_name = ?', (user_name,))
        conn.commit()

def delete_all_records():
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM comparative_evals')
        conn.commit()

def get_finished_user_count(total_required):
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM (
                SELECT user_name
                FROM comparative_evals
                GROUP BY user_name
                HAVING COUNT(DISTINCT query_file) >= ?
            )
        ''', (total_required,))
        return cursor.fetchone()[0]

def get_all_results():
    with sqlite3.connect(DB_FILE, timeout=15) as conn:
        return pd.read_sql_query("SELECT * FROM comparative_evals", conn)

# --- 5. File Loading ---
def load_files():
    files_q = sorted([f for f in os.listdir(QUERY_DIR) if f.endswith('.txt')])
    files_a = sorted([f for f in os.listdir(SOURCE_A_DIR) if f.endswith('.txt')])
    files_b = sorted([f for f in os.listdir(SOURCE_B_DIR) if f.endswith('.txt')])
    if not (len(files_q) == len(files_a) == len(files_b)):
        st.error(f"File count mismatch! Q:{len(files_q)}, A:{len(files_a)}, B:{len(files_b)}")
        st.stop()
    return files_q, files_a, files_b

init_db()

# --- 6. User Sidebar Controls ---
with st.sidebar:
    try:
        f_q, _, _ = load_files()
        total_queries = len(f_q)
        finished_count = get_finished_user_count(total_queries)
        st.metric("Total Users Finished", finished_count)
    except:
        pass

    if st.session_state.user_name:
        st.header(f"👤 {st.session_state.user_name}")

        # Navigation for Admin
        if st.session_state.user_name == "Admin":
            st.subheader("Admin Navigation")
            if st.button("📊 View Results Dashboard"):
                st.session_state.view_mode = "results"
                st.rerun()
            if st.button("📝 Back to Evaluation"):
                st.session_state.view_mode = "eval"
                st.rerun()

            st.subheader("Database Management")
            confirm_wipe = st.checkbox("Confirm Wipe All Data")
            if st.button("🔥 DELETE ALL RECORDS", disabled=not confirm_wipe):
                delete_all_records()
                st.session_state.user_name = ""
                st.session_state.current_pair = 0
                st.success("Database wiped!")
                st.rerun()

        # Updated Reset Progress logic
        st.markdown("---")
        if st.button(f"🚨 Reset My Progress"):
            reset_user_ratings(st.session_state.user_name)
            # CLEARING THE NAME SO IT GOES TO LOGIN
            st.session_state.user_name = ""
            st.session_state.current_pair = 0
            st.session_state.view_mode = "eval"
            st.rerun()
    else:
        st.info("Please log in.")

# --- 7. Main UI Logic ---

# Scenario A: Login Screen
if not st.session_state.user_name:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Query Answer Evaluation")
        name_input = st.text_input("Enter your name:")
        if st.button("Start"):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a valid name.")

# Scenario B: Admin Dashboard View
elif st.session_state.view_mode == "results" and st.session_state.user_name == "Admin":
    st.title("Admin Results Dashboard")
    df = get_all_results()

    if df.empty:
        st.info("No records found in the database.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Responses", len(df))
        m2.metric("Unique Evaluators", df['user_name'].nunique())

        st.markdown("### Selection History")
        st.dataframe(df, width="stretch")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "evaluation_results.csv", "text/csv")

# Scenario C: Evaluation View (Standard)
else:
    st.title(f"Evaluator: {st.session_state.user_name}")

    files_q, files_a, files_b = load_files()
    total_evals = len(files_a)

    progress_val = st.session_state.current_pair / total_evals if total_evals > 0 else 0
    current_eval = min(st.session_state.current_pair + 1, total_evals)
    st.progress(progress_val)
    st.write(f"**Task {current_eval} of {total_evals}**")

    if st.session_state.current_pair >= total_evals:
        st.success("You have completed all evaluations. Thank you!")
        if st.button("Finish & Log Out"):
            st.session_state.user_name = ""
            st.session_state.current_pair = 0
            st.rerun()
    else:
        with st.expander("📚 **How to judge Creativity (The NUS Framework)**", expanded=True):
            st.markdown("""
            **Creativity is defined as a combination of three elements:**
            * 🌟 **New (Novelty & Rarity):** Is the answer a genuinely new invention or a fresh synthesis that avoids tropes or cliches?
            * 🛠️ **Useful (Viability & Alignment):** Is the answer actionable, logically sound, and aligned with the query's constraints?
            * 🤯 **Surprising (Subversion & Trajectory):** Did the answer take a clever, non-obvious, or unpredictable lateral leap?
            """)

        idx = st.session_state.current_pair
        query_name = files_q[idx]
        file_a_name = files_a[idx]
        file_b_name = files_b[idx]

        with open(os.path.join(QUERY_DIR, query_name), 'r', encoding='utf-8') as f:
            text_q = f.read()
        with open(os.path.join(SOURCE_A_DIR, file_a_name), 'r', encoding='utf-8') as f:
            text_a = f.read()
        with open(os.path.join(SOURCE_B_DIR, file_b_name), 'r', encoding='utf-8') as f:
            text_b = f.read()

        st.markdown("### Query")
        st.info(text_q)
        st.markdown("---")

        seed_string = f"{st.session_state.user_name}_{idx}"
        rng = random.Random(seed_string)
        swap = rng.choice([True, False])

        col1, col2 = st.columns(2)
        left_text, right_text = (text_b, text_a) if swap else (text_a, text_b)
        left_name, right_name = (file_b_name, file_a_name) if swap else (file_a_name, file_b_name)

        with col1:
            st.subheader("Answer 1")
            st.markdown(left_text)

        with col2:
            st.subheader("Answer 2")
            st.markdown(right_text)

        st.markdown("---")
        st.markdown("### Which answer is more creative?")

        choice = st.radio(
            "Based on the New, Useful, and Surprising (NUS) metrics:",
            ["Answer 1", "Answer 2"],
            index=None,
            horizontal=True,
            key=f"choice_{idx}"
        )

        if st.button("Submit Choice & Continue", type="primary"):
            if choice is None:
                st.error("⚠️ Please select a winner before continuing.")
            else:
                winning_file = left_name if choice == "Answer 1" else right_name
                save_result(query_name, left_name, right_name, st.session_state.user_name, choice, winning_file)
                st.session_state.current_pair += 1
                st.rerun()
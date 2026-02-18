import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CONFIG

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mental Health Assessment", layout="centered")

# SIDEBAR

page = st.sidebar.radio(
    "Navigation",
    [
        "Run AI Assessment",
        "PHQ-9 Questionnaire",
        "Multi-Domain Assessment",
        "Patient History Dashboard"
    ]
)


# PAGE 1 — AI

if page == "Run AI Assessment":

    st.title("AI Depression Assessment")

    user_id = st.text_input("Patient ID")
    patient_text = st.text_area("How have you been feeling recently?")
    relative_text = st.text_area("Observed behavioral changes")

    if st.button("Run AI Assessment"):

        if not user_id or not patient_text:
            st.error("Patient ID and text input required.")
        else:
            response = requests.post(
                f"{BASE_URL}/assessments/create",
                json={
                    "user_id": user_id,
                    "patient_text": patient_text,
                    "relative_text": relative_text
                }
            )

            result = response.json()

            st.success("Assessment Complete")
            st.metric("AI Score", f"{result['score']} / 10")
            st.metric("Risk Level", result["level"])


# PAGE 2 — PHQ9

elif page == "PHQ-9 Questionnaire":

    st.title("PHQ-9 Questionnaire")

    user_id = st.text_input("Patient ID")

    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down ,depressed,or hopeless?",
        "Trouble falling/staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feel bad about yourself or that you are a failure or have let yourself or your family down?",
        "Have trouble concentrating on things, such as reading, work, or watching television?",
        "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
        "Thoughts you’d be better off dead or hurting yourself?"
    ]

    options = ["Not at all (0)", "Several days (1)",
               "More than half (2)", "Nearly every day (3)"]

    answers = {}
    for i, q in enumerate(questions):
        choice = st.radio(q, options, key=f"phq_{i}")
        answers[f"q{i+1}"] = options.index(choice)

    if st.button("Submit PHQ-9"):

        if not user_id:
            st.error("Patient ID required.")
        else:
            response = requests.post(
                f"{BASE_URL}/phq9/submit",
                json={"user_id": user_id, "answers": answers}
            )
            result = response.json()

            st.success("Submitted")
            st.metric("PHQ-9 Score", f"{result['score']} / 10")
            st.metric("Risk Level", result["level"])


# PAGE 3 — MULTI-DOMAIN

elif page == "Multi-Domain Assessment":

    st.title(" Multi-Domain Screening")

    user_id = st.text_input("Patient ID")

    module = st.selectbox(
        "Select Module",
        ["anxiety", "self_esteem", "procrastination",
         "workstress", "trauma", "grief",
         "relationship", "anger", "mental_health"]
    )

    question_bank = {
        "anxiety": [
            "Feeling nervous, anxious, or on edge?",
            "Not being able to stop or control worrying?",
            "Worrying too much about different things?",
            "Trouble relaxing?",
            "Being so restless that it's hard to sit still?",
            "Becoming easily annoyed or irritable?",
            "Feeling afraid as if something awful might happen?"
        ],
        "self_esteem": [
            "I feel confident in my abilities.",
            "I feel that I am a person of worth.",
            "I often feel useless or inadequate.",
            "I compare myself negatively to others.",
            "I feel proud of achievements.",
            "I doubt myself even when I succeed."
        ],
        "procrastination": [
            "I delay starting important tasks.",
            "I wait until the last minute to complete assignments.",
            "I feel overwhelmed and avoid responsibilities.",
            "Distract yourself?",
            "Regret delaying?"
        ],
        "workstress": [
            "Overwhelmed?",
            "Deadline pressure?",
            "Emotionally exhausted?",
            "Performance pressure?",
            "Job insecurity?"
        ],
        "trauma": [
            "Have you experienced a distressing or traumatic event?",
            "Do you have intrusive memories about it?",
            "Do you avoid reminders of the event?",
            "Do you feel emotionally numb?",
            "Do you feel constantly on guard or easily startled?"
        ],
        "grief": [
            "Intense sadness?",
            "Difficulty accepting loss?",
            "Life feels meaningless?",
            "Avoid reminders?",
            "Feel guilt?"
        ],
        "relationship": [
            "I feel emotionally supported in my relationship.",
            "Feel misunderstood?",
            "Communication with my partner/family is difficult.",
            "Relationship anxiety?",
            "Conflicts remain unresolved for long periods."
        ],
        "anger": [
            "Lose temper easily?",
            "Feel intense anger over small issues.",
            "Regret anger?",
            "Hard to calm down?",
            "Does your anger affect your relationships, work, or daily life?"
        ],
        "mental_health": [
            "Feel emotionally stable?",
            "Motivated daily?",
            "Hopeful about future?",
            "Struggle to enjoy activities?",
            "Overwhelmed emotionally?",
            "Socially connected?"
        ]
    }

    # Reverse scoring map
    reverse_items = {
        "self_esteem": [0, 1],
        "mental_health": [0, 1, 2, 5],
        "relationship": [0]
    }

    options = ["Not at all (0)", "Several days (1)",
               "More than half the time (2)", "Nearly every day (3)"]

    answers = {}
    for i, q in enumerate(question_bank[module]):
        choice = st.radio(q, options, key=f"{module}_{i}")
        answers[f"q{i+1}"] = options.index(choice)

    if st.button("Submit Module"):

        if not user_id:
            st.error("Patient ID required.")
        else:

            # Apply reverse scoring
            if module in reverse_items:
                for idx in reverse_items[module]:
                    key = f"q{idx+1}"
                    answers[key] = 3 - answers[key]

            response = requests.post(
                f"{BASE_URL}/questionnaire/submit",
                json={
                    "user_id": user_id,
                    "module": module,
                    "answers": answers
                }
            )

            result = response.json()

            st.success("Submitted")
            st.metric(f"{module.upper()} Score",
                      f"{result['score']} / 10")
            st.metric("Risk Level", result["level"])


# PAGE 4 — DASHBOARD

elif page == "Patient History Dashboard":

    st.title("📊 Patient Dashboard")

    user_id = st.text_input("Enter Patient ID")

    if not user_id:
        st.info("Please enter a Patient ID.")
        st.stop()

   
    # FETCH DATA
    
    try:
        tests_response = requests.get(f"{BASE_URL}/tests/{user_id}")
        sessions_response = requests.get(f"{BASE_URL}/sessions/{user_id}")

        tests_data = tests_response.json()
        sessions_data = sessions_response.json()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if not tests_data:
        st.warning("No test records found.")
        st.stop()

    
    # CREATE DATAFRAMES
    
    tests_df = pd.DataFrame(tests_data)
    tests_df["created_at"] = pd.to_datetime(tests_df["created_at"])
    tests_df = tests_df.sort_values("created_at")

    sessions_df = pd.DataFrame(sessions_data)

    if not sessions_df.empty:
        sessions_df["created_at"] = pd.to_datetime(sessions_df["created_at"])
        sessions_df = sessions_df.sort_values("created_at")
    
    #select session
    
    if not sessions_df.empty:

        session_options = sessions_df["session_number"].tolist()

        selected_session = st.selectbox(
            "Select Session",
            session_options
        )

        selected_session_data = sessions_df[
            sessions_df["session_number"] == selected_session
        ].iloc[0]

        st.metric(
            "Selected Session Score",
            f"{selected_session_data['session_score']} / 10"
        )

        session_id = selected_session_data["session_id"]

        session_tests = tests_df[
            tests_df["session_id"] == session_id
        ]

        st.subheader("Tests in Selected Session")
        st.dataframe(session_tests)

        st.markdown("---")
        st.subheader("Compare Sessions")

        compare_sessions = st.multiselect(
            "Select Two Sessions to Compare",
            sessions_df["session_number"].tolist()
        )

        if len(compare_sessions) == 2:

            s1 = sessions_df[
                sessions_df["session_number"] == compare_sessions[0]
            ].iloc[0]

            s2 = sessions_df[
                sessions_df["session_number"] == compare_sessions[1]
            ].iloc[0]

            diff = round(
                s2["session_score"] - s1["session_score"],
                2
            )

            st.metric("Session Score Difference", diff)

            if diff > 0:
                st.warning("Risk increased.")
            elif diff < 0:
                st.success("Improvement detected.")
            else:
                st.info("No change.")


    
    # SESSION OVERVIEW
    
    st.subheader("Sessions Overview")

    if not sessions_df.empty:
        st.dataframe(sessions_df)

        fig_session, ax_session = plt.subplots()
        ax_session.plot(
            sessions_df["created_at"],
            sessions_df["session_score"],
            marker="o"
        )
        ax_session.set_title("Session Score Over Time")
        ax_session.set_xlabel("Date")
        ax_session.set_ylabel("Weighted Score (/10)")
        plt.xticks(rotation=45)
        st.pyplot(fig_session)
    else:
        st.info("No sessions found.")

    st.markdown("---")

    
    # TESTS GROUPED BY SESSION
    
    st.subheader("Tests Grouped by Session")

    if not sessions_df.empty:

        for _, session_row in sessions_df.iterrows():

            session_id = session_row["session_id"]
            session_score = session_row["session_score"]

            st.markdown(f"### Session {session_id}")

            session_tests = tests_df[
                tests_df["session_id"] == session_id
            ]

            if not session_tests.empty:
                st.dataframe(
                    session_tests[
                        ["test_type", "score", "level", "created_at"]
                    ]
                )

                st.metric(
                    "Session Weighted Score",
                    f"{session_score} / 10"
                )

            st.markdown("---")

    else:
        st.info("No test sessions available.")

    
    # RADAR PROFILE (Latest per module)
    
    st.subheader("Latest Mental Health Profile")

    latest = tests_df.sort_values("created_at").groupby(
        "test_type"
    ).last()

    categories = latest.index.tolist()
    values = latest["score"].tolist()

    if len(values) >= 3:
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        st.pyplot(fig)
    else:
        st.info("Not enough modules for radar chart.")



# FOOTER

st.markdown("---")
st.caption("⚠️ Academic and research tool only. Not a medical diagnosis.")

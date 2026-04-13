import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="IPL Performance Analytics & Match Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Performance Analytics & Match Outcome Predictor")
st.write("Analyze IPL match data, team performance, top players, and predict match winners.")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# -------------------------------
# DATA CLEANING
# -------------------------------
def clean_matches(df):
    df = df.copy()

    # Drop unnecessary columns only if they exist
    drop_cols = ['method', 'umpire1', 'umpire2']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Fill missing values
    if 'city' in df.columns:
        df['city'] = df['city'].fillna('Unknown')
    if 'player_of_match' in df.columns:
        df['player_of_match'] = df['player_of_match'].fillna('Unknown')
    if 'result_margin' in df.columns:
        df['result_margin'] = df['result_margin'].fillna(0)

    # Date parsing
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['season_year'] = df['date'].dt.year
    elif 'season' in df.columns:
        df['season_year'] = df['season']

    # Remove rows without winner
    if 'winner' in df.columns:
        df = df.dropna(subset=['winner'])

    # Toss win and match win feature
    if 'toss_winner' in df.columns and 'winner' in df.columns:
        df['toss_win_match'] = (df['toss_winner'] == df['winner']).astype(int)

    return df

matches = clean_matches(matches)

# -------------------------------
# TEAM NAME STANDARDIZATION
# -------------------------------
team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants'
}

def standardize_team_names(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(team_name_map)
    return df

match_team_cols = ['team1', 'team2', 'winner', 'toss_winner']
matches = standardize_team_names(matches, match_team_cols)

delivery_team_cols = ['batting_team', 'bowling_team']
deliveries = standardize_team_names(deliveries, delivery_team_cols)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Filter Options")

teams = sorted(pd.concat([matches['team1'], matches['team2']]).dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Select Team", ["All Teams"] + teams)

seasons = []
if 'season_year' in matches.columns:
    seasons = sorted(matches['season_year'].dropna().unique().tolist())
selected_season = st.sidebar.selectbox("Select Season", ["All Seasons"] + [int(s) for s in seasons])

filtered_matches = matches.copy()

if selected_team != "All Teams":
    filtered_matches = filtered_matches[
        (filtered_matches['team1'] == selected_team) |
        (filtered_matches['team2'] == selected_team)
    ]

if selected_season != "All Seasons":
    filtered_matches = filtered_matches[filtered_matches['season_year'] == selected_season]

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total_matches = len(filtered_matches)
total_teams = len(pd.concat([filtered_matches['team1'], filtered_matches['team2']]).dropna().unique())
total_cities = filtered_matches['city'].nunique() if 'city' in filtered_matches.columns else 0
total_seasons = filtered_matches['season_year'].nunique() if 'season_year' in filtered_matches.columns else 0

col1.metric("Total Matches", total_matches)
col2.metric("Teams", total_teams)
col3.metric("Cities", total_cities)
col4.metric("Seasons", total_seasons)

# -------------------------------
# TEAM WINS ANALYSIS
# -------------------------------
st.subheader("🏆 Team Wins Analysis")

team_wins = filtered_matches['winner'].value_counts().reset_index()
team_wins.columns = ['Team', 'Wins']

if not team_wins.empty:
    fig_wins = px.bar(
        team_wins,
        x='Team',
        y='Wins',
        title='Team-wise Match Wins'
    )
    st.plotly_chart(fig_wins, use_container_width=True)
else:
    st.warning("No data available for selected filters.")

# -------------------------------
# TOSS IMPACT
# -------------------------------
st.subheader("🪙 Toss Impact on Match Result")

if 'toss_win_match' in filtered_matches.columns:
    toss_impact = filtered_matches['toss_win_match'].value_counts().reset_index()
    toss_impact.columns = ['Toss Winner Also Won Match', 'Count']
    toss_impact['Toss Winner Also Won Match'] = toss_impact['Toss Winner Also Won Match'].map({1: 'Yes', 0: 'No'})

    fig_toss = px.pie(
        toss_impact,
        names='Toss Winner Also Won Match',
        values='Count',
        title='Toss Win vs Match Win'
    )
    st.plotly_chart(fig_toss, use_container_width=True)

# -------------------------------
# PLAYER PERFORMANCE
# -------------------------------
st.subheader("🏏 Top Batters")

# Batter stats
batter_runs = deliveries.groupby('batter', as_index=False)['batsman_runs'].sum()
# Batter stats
batter_runs = deliveries.groupby('batter', as_index=False)['batsman_runs'].sum()

# FIXED (no error)
balls_faced = deliveries.groupby('batter').size().reset_index(name='balls_faced')

batting_stats = batter_runs.merge(balls_faced, on='batter', how='left')

batting_stats['strike_rate'] = np.where(
    batting_stats['balls_faced'] > 0,
    (batting_stats['batsman_runs'] / batting_stats['balls_faced']) * 100,
    0
)
batting_stats['batting_score'] = (batting_stats['batsman_runs'] * 0.6) + (batting_stats['strike_rate'] * 0.4)
top_batters = batting_stats.sort_values(by='batting_score', ascending=False).head(10)

fig_batters = px.bar(
    top_batters,
    x='batter',
    y='batting_score',
    title='Top 10 Batters by Performance Score'
)
st.plotly_chart(fig_batters, use_container_width=True)

st.subheader("🎯 Top Bowlers")

# Bowler stats
wicket_kinds = ['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket']
wickets_df = deliveries[deliveries['dismissal_kind'].isin(wicket_kinds)]
bowler_wickets = wickets_df.groupby('bowler', as_index=False).size()
bowler_wickets.columns = ['bowler', 'wickets']

top_bowlers = bowler_wickets.sort_values(by='wickets', ascending=False).head(10)

fig_bowlers = px.bar(
    top_bowlers,
    x='bowler',
    y='wickets',
    title='Top 10 Bowlers by Wickets'
)
st.plotly_chart(fig_bowlers, use_container_width=True)

# -------------------------------
# PLAYER VALUE RANKER
# -------------------------------
st.subheader("⭐ Player Value Ranker")

player_value = batting_stats[['batter', 'batsman_runs', 'strike_rate', 'batting_score']].rename(
    columns={'batter': 'player'}
)

bowler_value = bowler_wickets.rename(columns={'bowler': 'player'})
player_value = player_value.merge(bowler_value, on='player', how='outer')
player_value['batsman_runs'] = player_value['batsman_runs'].fillna(0)
player_value['strike_rate'] = player_value['strike_rate'].fillna(0)
player_value['batting_score'] = player_value['batting_score'].fillna(0)
player_value['wickets'] = player_value['wickets'].fillna(0)

player_value['overall_value'] = player_value['batting_score'] + (player_value['wickets'] * 10)
top_players = player_value.sort_values(by='overall_value', ascending=False).head(15)

st.dataframe(top_players, use_container_width=True)

# -------------------------------
# MATCH WINNER PREDICTION MODEL
# -------------------------------
st.subheader("🤖 Match Winner Prediction Model")

model_df = matches.copy()

required_cols = ['team1', 'team2', 'toss_winner', 'city', 'winner']
model_df = model_df.dropna(subset=[col for col in required_cols if col in model_df.columns])

# Keep only rows where winner is one of team1/team2
model_df = model_df[
    (model_df['winner'] == model_df['team1']) |
    (model_df['winner'] == model_df['team2'])
].copy()

# Target variable: 1 if team1 wins else 0
model_df['team1_win'] = (model_df['winner'] == model_df['team1']).astype(int)

feature_cols = ['team1', 'team2', 'toss_winner', 'city']
X = model_df[feature_cols]
y = model_df['team1_win']

categorical_features = ['team1', 'team2', 'toss_winner', 'city']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy:.2%}")

with st.expander("See Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.subheader("🔮 Predict Match Winner")

predict_team1 = st.selectbox("Select Team 1", teams, key="pred_team1")
predict_team2 = st.selectbox("Select Team 2", teams, index=1 if len(teams) > 1 else 0, key="pred_team2")
predict_toss_winner = st.selectbox("Select Toss Winner", teams, key="pred_toss")
predict_city = st.selectbox("Select City", sorted(matches['city'].dropna().unique().tolist()), key="pred_city")

if st.button("Predict Winner"):
    if predict_team1 == predict_team2:
        st.error("Team 1 and Team 2 must be different.")
    else:
        input_df = pd.DataFrame([{
            'team1': predict_team1,
            'team2': predict_team2,
            'toss_winner': predict_toss_winner,
            'city': predict_city
        }])

        pred = model_pipeline.predict(input_df)[0]
        proba = model_pipeline.predict_proba(input_df)[0]

        predicted_winner = predict_team1 if pred == 1 else predict_team2
        team1_prob = proba[1] if len(proba) > 1 else 0.5
        team2_prob = proba[0] if len(proba) > 1 else 0.5

        st.success(f"Predicted Winner: {predicted_winner}")
        st.write(f"**{predict_team1} Win Probability:** {team1_prob:.2%}")
        st.write(f"**{predict_team2} Win Probability:** {team2_prob:.2%}")

# -------------------------------
# RAW DATA
# -------------------------------
with st.expander("View Raw Match Data"):
    st.dataframe(filtered_matches.head(100), use_container_width=True)

with st.expander("View Raw Deliveries Data"):
    st.dataframe(deliveries.head(100), use_container_width=True)
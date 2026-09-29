"""Synthesizes the Student Performance Factors schema (19 attrs + Exam_Score),
calibrated to a Nigerian polytechnic cohort. Drop the real Kaggle CSV at
data/StudentPerformanceFactors.csv and retrain to use it instead."""
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42); N = 6607
def choice(o, p): return rng.choice(o, size=N, p=p)
df = pd.DataFrame({
 "Hours_Studied": np.clip(rng.normal(20,6,N),1,44).round().astype(int),
 "Attendance": np.clip(rng.normal(80,11,N),60,100).round().astype(int),
 "Parental_Involvement": choice(["Low","Medium","High"],[.25,.5,.25]),
 "Access_to_Resources": choice(["Low","Medium","High"],[.3,.45,.25]),
 "Extracurricular_Activities": choice(["No","Yes"],[.42,.58]),
 "Sleep_Hours": np.clip(rng.normal(7,1.4,N),4,10).round().astype(int),
 "Previous_Scores": np.clip(rng.normal(72,13,N),40,100).round().astype(int),
 "Motivation_Level": choice(["Low","Medium","High"],[.28,.48,.24]),
 "Internet_Access": choice(["No","Yes"],[.18,.82]),
 "Tutoring_Sessions": np.clip(rng.poisson(1.5,N),0,8).astype(int),
 "Family_Income": choice(["Low","Medium","High"],[.4,.4,.2]),
 "Teacher_Quality": choice(["Low","Medium","High"],[.2,.55,.25]),
 "School_Type": choice(["Public","Private"],[.69,.31]),
 "Peer_Influence": choice(["Negative","Neutral","Positive"],[.22,.38,.4]),
 "Physical_Activity": np.clip(rng.normal(3,1.2,N),0,6).round().astype(int),
 "Learning_Disabilities": choice(["No","Yes"],[.89,.11]),
 "Parental_Education_Level": choice(["High School","College","Postgraduate"],[.45,.38,.17]),
 "Distance_from_Home": choice(["Near","Moderate","Far"],[.55,.3,.15]),
 "Gender": choice(["Male","Female"],[.55,.45]),
})
lvl={"Low":0,"Medium":1,"High":2}; peer={"Negative":-1,"Neutral":0,"Positive":1}
pedu={"High School":0,"College":1,"Postgraduate":2}; dist={"Near":1,"Moderate":0,"Far":-1}
score=(18+0.90*df.Hours_Studied+0.35*df.Attendance+0.32*df.Previous_Scores
 +1.6*df.Motivation_Level.map(lvl)+1.5*df.Access_to_Resources.map(lvl)
 +1.3*df.Parental_Involvement.map(lvl)+1.1*df.Tutoring_Sessions
 +1.0*df.Teacher_Quality.map(lvl)+1.2*df.Peer_Influence.map(peer)
 +0.8*df.Parental_Education_Level.map(pedu)+0.9*df.Internet_Access.map({"No":-1,"Yes":1})
 +0.7*df.Family_Income.map(lvl)+0.6*df.Distance_from_Home.map(dist)
 -0.5*(df.Sleep_Hours-7).abs()+0.4*df.Physical_Activity
 -4.5*df.Learning_Disabilities.map({"No":0,"Yes":1})+rng.normal(0,0.5,N))
score = 57 + (score-score.mean())/score.std()*12   # Nigerian polytechnic distribution
df["Exam_Score"] = np.clip(score,20,100).round().astype(int)
for col in ["Teacher_Quality","Parental_Education_Level","Distance_from_Home"]:
    df.loc[rng.random(N)<0.012, col] = np.nan
out = Path(__file__).resolve().parents[1]/"data"/"StudentPerformanceFactors.csv"
df.to_csv(out, index=False); print(f"Wrote {len(df)} -> {out}")

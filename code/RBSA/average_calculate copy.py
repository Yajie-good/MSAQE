import pandas as pd


# Read data


# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/bj_yiheyuan_score.csv"
# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gx_lj_score.csv"
# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gz_as_score.csv"
# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/hb_sxrj_score.csv"
# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/jilin_changying_score.csv"
# file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/shaanxi_dtfry_score.csv"
file_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/zj_xt_score.csv"

df = pd.read_csv(file_path)

# Calculate the average of each column
avg_scores = {
    "BMscore_avg": df["BMscore"].mean(),
    "Excscore_avg": df["Excscore"].mean(),
    "Hygscore_avg": df["Hygscore"].mean(),
    "PTscore_avg": df["PTscore"].mean(),
    "REPscore_avg": df["REPscore"].mean(),
    "TSAscore_avg": df["TSAscore"].mean(),
    "TSHscore_avg": df["TSHscore"].mean(),
    "TTscore_avg": df["TTscore"].mean()
}

# Convert results to DataFrame
avg_df = pd.DataFrame([avg_scores])


# Save results
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/bj_yiheyuan_score_avg.csv"
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gx_lj_score_avg.csv"
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gz_as_score_avg.csv"
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/hb_sxrj_score_avg.csv"
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/jilin_changying_score_avg.csv"
# save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/shaanxi_dtfry_score_avg.csv"
save_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/zj_xt_score_avg.csv"

avg_df.to_csv(save_path, index=False)

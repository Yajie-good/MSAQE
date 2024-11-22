import pandas as pd



# Define lists of input and output file paths
score_files = [
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/bj_yiheyuan_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/gx_lj_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/gz_as_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/hb_sxrj_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/jilin_changying_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/shaanxi_dtfry_classification_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score/zj_xt_classification_score.csv"
]

sentiment_files = [
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/bj_yiheyuan_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gx_lj_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gz_as_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/hb_sxrj_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/jilin_changying_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/shaanxi_dtfry_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/zj_xt_sentiment_final_albert.csv"
]

output_files = [
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/bj_yiheyuan_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/gx_lj_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/gz_as_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/hb_sxrj_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/jilin_changying_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/shaanxi_dtfry_classification_final_score.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_final/zj_xt_classification_final_score.csv"
]

# Loop through each file pair
for score_file, sentiment_file, output_file in zip(score_files, sentiment_files, output_files):
    # Read sentiment score file and score file
    score_df = pd.read_csv(score_file)
    sentiment_df = pd.read_csv(sentiment_file)

    # Extract necessary columns
    description_column = score_df['description']
    sentiment_column = sentiment_df['sentiment']

    # Calculate final score with weights
    for column in ['BMscore', 'Excscore', 'Hygscore', 'PTscore', 'REPscore', 'TSAscore', 'TSHscore', 'TTscore']:
        score_df[column] = score_df[column] * 0.5 + sentiment_column * 0.5

    # Save description column and updated score columns to a new file
    final_df = pd.DataFrame({
        'description': description_column,
        'BMscore': score_df['BMscore'],
        'Excscore': score_df['Excscore'],
        'Hygscore': score_df['Hygscore'],
        'PTscore': score_df['PTscore'],
        'REPscore': score_df['REPscore'],
        'TSAscore': score_df['TSAscore'],
        'TSHscore': score_df['TSHscore'],
        'TTscore': score_df['TTscore']
    })
    
    final_df.to_csv(output_file, index=False)
    print(f"Final sentiment score results saved to {output_file}")
            







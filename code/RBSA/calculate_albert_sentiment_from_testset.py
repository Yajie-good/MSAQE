# 1. Calculate sentiment scores for descriptions in test_samples using ALBERT


# import os
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # # Set environment variables to make only specified GPU visible
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # # Check for available GPU and use the first available GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # # Specify the local path of the model
# model_path = "/data/0WYJ/newdata_wyj/CMLTES_codes/reference_test/sentiment/"

# # # Load model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Define lists of input and output file paths
# input_files = [
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/bj_yiheyuan.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gx_lj.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gz_as.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/hb_sxrj.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/jilin_changying.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/zj_xt.csv"
# ]

# output_files = [
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/bj_yiheyuan_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gx_lj_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gz_as_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/hb_sxrj_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/jilin_changying_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/shaanxi_dtfry_sentiment_albert.csv",
#     "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/zj_xt_sentiment_albert.csv"
# ]

# # Loop through each file, calculate sentiment scores, and save results
# for input_file, output_file in zip(input_files, output_files):
#     # Read data
#     df = pd.read_csv(input_file)

#     # Create a new column to store sentiment analysis results
#     sentiment_scores = []

#     # Iterate over each row to calculate sentiment scores
#     for _, row in df.iterrows():
#         description = row['description']
#         tokens = tokenizer.encode(description, return_tensors='pt', truncation=True, max_length=510).to(device)
#         result = model(tokens)[0]
#         sentiment_score = result.detach().cpu().tolist()
#         sentiment_scores.append(sentiment_score)


#     # Add sentiment scores to DataFrame
#     df['sentiment'] = sentiment_scores

#     # Save new data to output file
#     df.to_csv(output_file, index=False)
#     print(f"Final sentiment analysis results saved to {output_file}")

#2、Calculate final sentiment score from sentiment values obtained in the previous step
#If both values in the sentiment column are positive, label as positive and take the average score.
# If both values are negative, label as negative and take the average score.
# If one is positive and one is negative, use the value with the higher absolute value as the score and label based on its sign.
import os
import pandas as pd
import ast

# Define lists of input and output file paths
input_files = [
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/bj_yiheyuan_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gx_lj_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gz_as_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/hb_sxrj_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/jilin_changying_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/shaanxi_dtfry_sentiment_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/zj_xt_sentiment_albert.csv"
]

output_files = [
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/bj_yiheyuan_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gx_lj_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/gz_as_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/hb_sxrj_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/jilin_changying_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/shaanxi_dtfry_sentiment_final_albert.csv",
    "/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_classification_score_albert/zj_xt_sentiment_final_albert.csv"
]

# Loop through each file to calculate the final sentiment score
for input_file, output_file in zip(input_files, output_files):
    # Read data
    df = pd.read_csv(input_file)

    # Create a new list to store the calculated sentiment scores
    final_sentiments = []

    for _, row in df.iterrows():
        # Get sentiment scores
        sentiment_scores = ast.literal_eval(row['sentiment'])[0]

        # If both values are positive
        if sentiment_scores[0] > 0 and sentiment_scores[1] > 0:
            sentiment_label = '正面'
            score = (sentiment_scores[0] + sentiment_scores[1]) / 2  # 取两个数的平均

        # If both values are negative
        elif sentiment_scores[0] < 0 and sentiment_scores[1] < 0:
            sentiment_label = '负面'
            score = (sentiment_scores[0] + sentiment_scores[1]) / 2  # 取两个数的平均

        # If one value is positive and the other is negative
        else:
            if abs(sentiment_scores[0]) > abs(sentiment_scores[1]):
                sentiment_label = '正面' if sentiment_scores[0] > 0 else '负面'
                score = sentiment_scores[0]
            else:
                sentiment_label = '正面' if sentiment_scores[1] > 0 else '负面'
                score = sentiment_scores[1]

        # Append the calculated score to the list
        final_sentiments.append(score)

    # Add the calculated results to a new DataFrame
    df['sentiment'] = final_sentiments

    df[['description', 'sentiment']].to_csv(output_file, index=False)
    print(f"save sentiment result to {output_file}")

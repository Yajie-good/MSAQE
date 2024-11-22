# 1. First, convert Excel to CSV
# import pandas as pd

# Read the Excel file
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/bj_yiheyuan.xlsx'
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gx_lj.xlsx'
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gz_as.xlsx'
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/hb_sxrj.xlsx'
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/jilin_changying.xlsx'
# # excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.xlsx'
# excel_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/zj_xt.xlsx'
# df = pd.read_excel(excel_file_path)

# # # Save as a CSV file with UTF-8-SIG encoding
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/bj_yiheyuan.csv'
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gx_lj.csv'
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gz_as.csv'
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/hb_sxrj.csv'
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/jilin_changying.csv'
# # csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.csv'
# csv_file_path = '/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/zj_xt.csv'
# df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')


# # 2. Test weighted averaging method with single data entry, test single, reference single; test passed
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# from torch.nn import CosineSimilarity
# import torch
# import os

# # # Set GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# # # Function: Calculate semantic similarity between two texts
# def semantic_similarity(input_text, reference_text, model, tokenizer):
#     input_text_encoded = tokenizer(input_text, return_tensors='pt').to('cuda')
#     reference_text_encoded = tokenizer(reference_text, return_tensors='pt').to('cuda')

#     input_text_embedding = model(**input_text_encoded)[0].mean(dim=1).squeeze()
#     reference_text_embedding = model(**reference_text_encoded)[0].mean(dim=1).squeeze()

#     cosine_similarity = CosineSimilarity(dim=0)
#     similarity_score = cosine_similarity(input_text_embedding, reference_text_embedding)

#     return similarity_score.item()

# # # Load model and tokenizer
# model_directory = "/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/Bigbird_PRE"
# tokenizer = AutoTokenizer.from_pretrained(model_directory)
# model = AutoModel.from_pretrained(model_directory).to('cuda')

# # # Read the second row of the test dataset
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.csv").iloc[1:2]

# # # Prefix for reference datasets
# reference_prefixes = ["BM", "Exc", "Hyg", "PT", "REP", "TSA", "TSH", "TT"]

# # # Initialize list to store results
# results = []

# # # Use the second row of the test dataset for calculation
# test_text = test_df.iloc[0]["description"]
# reference_scores = {}

# # # Calculate semantic similarity between each reference dataset's second row and the test data
# for prefix in reference_prefixes:
#     for i in range(1, 3):
#         ref_df = pd.read_csv(f"/data/0WYJ/newdata_wyj/CMLTES_codes/reference_test/reference_{prefix}_sentiment_4_{i}.csv").iloc[1:2]
#         for _, ref_row in ref_df.iterrows():
#             similarity_score = semantic_similarity(test_text, ref_row["description"], model, tokenizer)
#             reference_scores[f"{prefix}score_{i}"] = similarity_score

# # # Add scores for the current test sample to the results list
# results.append({**{'description': test_text}, **reference_scores})

# # # Create DataFrame and save results
# final_scores_df = pd.DataFrame(results)

# # # Save results to CSV file
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/shaanxi_dtfry_score_test.csv", index=False, encoding='utf-8-sig')


# # 3. Calculate using the second row of the test dataset and the entire reference dataset for validation
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# from torch.nn import CosineSimilarity
# import torch
# import os

# # # Set GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# # # Load model and tokenizer
# model_directory = "/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/Bigbird_PRE"
# tokenizer = AutoTokenizer.from_pretrained(model_directory)
# model = AutoModel.from_pretrained(model_directory).to('cuda')

# # # Function: Calculate text embedding vector
# def get_embedding(text, model, tokenizer):
#     encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cuda')
#     with torch.no_grad():
#         embedding = model(**encoded)[0].mean(dim=1)
#     return embedding.squeeze()

# # # Function: Batch calculate semantic similarity
# def batch_semantic_similarity(input_embedding, reference_embeddings):
#     cosine_similarity = CosineSimilarity(dim=0)
#     similarity_scores = [cosine_similarity(input_embedding, ref_emb) for ref_emb in reference_embeddings]
#     return torch.stack(similarity_scores).cpu().numpy()

# # # Prefix for reference datasets
# reference_prefixes = ["BM", "Exc", "Hyg", "PT", "REP", "TSA", "TSH", "TT"]

# # # Precompute weighted average vectors for reference datasets
# reference_avg_embeddings = {}
# for prefix in reference_prefixes:
#     total_embedding = 0
#     total_weight = 0
#     for i in range(1, 3):
#         ref_df = pd.read_csv(f"/data/0WYJ/newdata_wyj/CMLTES_codes/reference_test/reference_{prefix}_sentiment_4_{i}.csv")
#         for _, row in ref_df.iterrows():
#             embedding = get_embedding(row['description'], model, tokenizer)
#             total_embedding += embedding * row['score']
#             total_weight += row['score']
#     avg_embedding = total_embedding / total_weight
#     reference_avg_embeddings[prefix] = avg_embedding

# # # Read the second row of the test dataset
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.csv").iloc[1:2]

# # # Use the second row of the test dataset for calculation
# test_text = test_df.iloc[0]["description"]
# test_embedding = get_embedding(test_text, model, tokenizer)
# reference_scores = {}

# # # Calculate the average score for each reference dataset
# for prefix, ref_avg_emb in reference_avg_embeddings.items():
#     similarity_score = batch_semantic_similarity(test_embedding, [ref_avg_emb])
#     reference_scores[f"{prefix}score"] = similarity_score[0]

# # # Create DataFrame and save results
# result_df = pd.DataFrame([{**{'description': test_text}, **reference_scores}])
# result_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/shaanxi_dtfry_score_test.csv", index=False, encoding='utf-8-sig')

#4. Calculate each row in the test dataset with all rows in the reference dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity
import torch
import os

# # Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # Load model and tokenizer
model_directory = "/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/Bigbird_PRE"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModel.from_pretrained(model_directory).to('cuda')

# # Function: Calculate text embedding vector
def get_embedding(text, model, tokenizer):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cuda')
    with torch.no_grad():
        embedding = model(**encoded)[0].mean(dim=1)
    return embedding.squeeze()

# # Function: Batch calculate semantic similarity
def batch_semantic_similarity(input_embedding, reference_embeddings):
    cosine_similarity = CosineSimilarity(dim=0)
    similarity_scores = [cosine_similarity(input_embedding, ref_emb) for ref_emb in reference_embeddings]
    return torch.stack(similarity_scores).cpu().numpy()

# # Prefix for reference datasets
reference_prefixes = ["BM", "Exc", "Hyg", "PT", "REP", "TSA", "TSH", "TT"]

# # Precompute weighted average vectors for reference datasets
reference_avg_embeddings = {}
for prefix in reference_prefixes:
    total_embedding = 0
    total_weight = 0
    for i in range(1, 3):
        ref_df = pd.read_csv(f"/data/0WYJ/newdata_wyj/CMLTES_codes/reference_test/reference_{prefix}_sentiment_4_{i}.csv")
        for _, row in ref_df.iterrows():
            embedding = get_embedding(row['description'], model, tokenizer)
            total_embedding += embedding * row['score']
            total_weight += row['score']
    avg_embedding = total_embedding / total_weight
    reference_avg_embeddings[prefix] = avg_embedding

# # Read the test dataset
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/bj_yiheyuan.csv")
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gx_lj.csv")
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/gz_as.csv")
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/hb_sxrj.csv")
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/jilin_changying.csv")
# test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/shaanxi_dtfry.csv")
test_df = pd.read_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples/zj_xt.csv")



# # Initialize DataFrame to store results
final_scores_df = pd.DataFrame()

# # Loop through the test dataset
for _, test_row in test_df.iterrows():
    test_text = test_row["description"]
    test_embedding = get_embedding(test_text, model, tokenizer)
    reference_scores = {}

    # # Calculate average score for each reference dataset
    for prefix, ref_avg_emb in reference_avg_embeddings.items():
        similarity_score = batch_semantic_similarity(test_embedding, [ref_avg_emb])
        reference_scores[f"{prefix}score"] = similarity_score[0]

    # # Add scores for the current test sample to the DataFrame
    new_row = pd.DataFrame([{**{'description': test_text}, **reference_scores}])
    final_scores_df = pd.concat([final_scores_df, new_row], ignore_index=True)

# # Save results to CSV file
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/bj_yiheyuan_score.csv", index=False, encoding='utf-8-sig')
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gx_lj_score.csv", index=False, encoding='utf-8-sig')
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/gz_as_score.csv", index=False, encoding='utf-8-sig')
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/hb_sxrj_score.csv", index=False, encoding='utf-8-sig')
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/jilin_changying_score.csv", index=False, encoding='utf-8-sig')
# final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/shaanxi_dtfry_score.csv", index=False, encoding='utf-8-sig')
final_scores_df.to_csv("/data/0WYJ/newdata_wyj/CMLTES_codes/data/score_test/test_samples_score/zj_xt_score.csv", index=False, encoding='utf-8-sig')
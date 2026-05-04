import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.utils import shannon_entropy, validate_features
from src.features import engineer_all_features, build_X_y, get_final_feature_list
from src.dl_model import train_model, DEVICE, TIER_NAMES

def main():
    # 1. Load and prepare training data
    print("Loading training data...")
    df_train = pd.read_csv('data/DNA Processed Data.csv')
    df_train_eng = engineer_all_features(df_train, verbose=False)
    X_train, y_train, train_features = build_X_y(df_train_eng)

    print("\nTraining MLP model...")
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model, history = train_model(
        X_tr.values, y_tr, X_vl.values, y_vl,
        epochs=50, verbose=1, batch_size=128
    )

    # 2. Load new dataset
    print("\nLoading new dataset...")
    df_new = pd.read_csv('data/new_users_dataset.csv')

    # 3. Map new dataset columns to expected schema
    print("Mapping columns...")
    df_mapped = pd.DataFrame()
    df_mapped['developer_id'] = df_new['login']
    df_mapped['account_age_days'] = df_new['account_age_days']
    df_mapped['total_commits'] = df_new['total_commit_count_estimate']
    df_mapped['commits_last_90d'] = df_new['active_days_last_90']
    df_mapped['total_repos'] = df_new['public_repos']
    df_mapped['pull_requests_merged'] = df_new['pr_review_events']
    df_mapped['issues_closed'] = df_new['issue_comment_events']
    df_mapped['stars_received'] = df_new['total_stars_received']
    df_mapped['languages_used'] = df_new['languages_list'].fillna('')
    df_mapped['has_readme_pct'] = df_new['has_readme_ratio']

    total_repos = df_new['forked_repos'] + df_new['original_repos']
    df_mapped['fork_to_original_ratio'] = np.where(total_repos > 0, df_new['forked_repos'] / total_repos, 0)
    
    df_mapped['years_active'] = df_new['account_age_years']
    df_mapped['has_bio'] = df_new['has_bio'].astype(int)
    df_mapped['has_company'] = df_new['has_company'].astype(int)
    df_mapped['has_location'] = df_new['has_location'].astype(int)
    df_mapped['has_blog'] = df_new['has_blog'].astype(int)
    df_mapped['bio_length'] = df_new['bio_length']
    df_mapped['avg_repo_size_kb'] = df_new['avg_repo_size_kb']

    # Language features
    def get_primary_language(langs):
        if not langs or pd.isna(langs): return 'Unknown'
        return str(langs).split(',')[0]

    df_mapped['primary_language'] = df_mapped['languages_used'].apply(get_primary_language)
    df_mapped['language_entropy'] = df_mapped['languages_used'].apply(
        lambda x: shannon_entropy(str(x).split(',')) if x else 0.0
    )

    # Fill missing columns expected by feature engineering with 0s
    df_mapped['framework_count'] = 0
    df_mapped['has_tests_pct'] = 0.0
    df_mapped['has_ci_pct'] = 0.0
    df_mapped['commit_message_avg_len'] = 0
    df_mapped['languages_per_repo_avg'] = 0.0
    df_mapped['commit_trend_slope'] = 0.0
    df_mapped['activity_decay_lambda'] = 0.0
    df_mapped['avg_repo_description_len'] = 0.0

    # 4. Engineer features for new data
    print("Engineering features for new data...")
    df_new_eng = engineer_all_features(df_mapped, verbose=False)
    
    # In case any new primary languages appear, the LabelEncoder from features.py 
    # creates new integers, which is fine as long as validate_features handles it.
    
    # 5. Validate features match training exactly
    print("Validating features...")
    X_new = validate_features(df_new_eng, train_features)

    # 6. Predict
    print("Generating predictions...")
    model.eval()
    X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_new_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    # 7. Save results
    df_new['predicted_tier_index'] = y_pred
    df_new['predicted_tier_name'] = [TIER_NAMES[idx] for idx in y_pred]

    out_path = 'data/new_users_predictions.csv'
    df_new.to_csv(out_path, index=False)
    
    print(f"\n✓ Predictions successfully saved to: {out_path}")
    print("\nPredicted Tier Distribution:")
    print(df_new['predicted_tier_name'].value_counts())

if __name__ == "__main__":
    main()

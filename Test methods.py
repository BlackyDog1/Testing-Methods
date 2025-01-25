# Test methods


def T_test_onnx_model_with_controls(X_test, main_feature, control_features=[]):
    predictions_1, predictions_2 = get_onnx_predictions(X_test)
    
    # Add control features to group by
    group_columns = control_features
    
    # Iterate over combinations of control features
    for control_values in X_test[group_columns].drop_duplicates().itertuples(index=False):
        # Filter by control feature values
        control_mask = (X_test[group_columns] == pd.Series(control_values, index=group_columns)).all(axis=1)
        control_group = X_test[control_mask]

        # Subset groups based on the main feature
        group_0 = control_group[control_group[main_feature] == 0]
        group_1 = control_group[control_group[main_feature] == 1]
        
        # Get predictions for these groups
        preds_0_model_1 = predictions_1[group_0.index]
        preds_1_model_1 = predictions_1[group_1.index]
        preds_0_model_2 = predictions_2[group_0.index]
        preds_1_model_2 = predictions_2[group_1.index]

        # Ensure there are enough samples in both groups
        if len(preds_0_model_1) > 1 and len(preds_1_model_1) > 1:
            # Perform T-test for model 1
            t_stat, p_value = ttest_ind(preds_0_model_1, preds_1_model_1, equal_var=False)
            print("  T-Test Results for Predictions in model 1:")
            print(f"    T-statistic: {t_stat}")
            print(f"    P-value: {p_value}")
            if p_value < 0.05:
                print("    Statistically significant difference in model 1 predictions.")

            # Perform T-test for model 2
            t_stat, p_value = ttest_ind(preds_0_model_2, preds_1_model_2, equal_var=False)
            print("  T-Test Results for Predictions in model 2:")
            print(f"    T-statistic: {t_stat}")
            print(f"    P-value: {p_value}")
            if p_value < 0.05:
                print("    Statistically significant difference in model 2 predictions.")


def controlled_subgroup_performance(X_test, main_feature, y_test, predictions_1, predictions_2, control_features=None):
    y_test = pd.Series(y_test).reset_index(drop=True)

    # Validate that control_features are in X_test
    if control_features is not None:
        for control_feature in control_features:
            if control_feature not in X_test.columns:
                print(f"Error: '{control_feature}' is not a valid column in X_test.")
                return

        # Get unique combinations of control feature values
        unique_combinations = X_test[control_features].drop_duplicates()

        # Iterate over combinations of control features
        for control_values in unique_combinations.itertuples(index=False):
            control_mask = (X_test[control_features] == pd.Series(control_values, index=control_features)).all(axis=1)
            control_group = X_test[control_mask]

            # Subset groups based on the main feature
            group_0 = control_group[control_group[main_feature] == 0]
            group_1 = control_group[control_group[main_feature] == 1]

            # Get predictions for these groups
            preds_0_model_1 = predictions_1[group_0.index]
            preds_1_model_1 = predictions_1[group_1.index]
            preds_0_model_2 = predictions_2[group_0.index]
            preds_1_model_2 = predictions_2[group_1.index]

            # Ensure there are enough samples in both groups
            print(f"Control group: {control_values}")
            print(f"  Group 0 size: {len(preds_0_model_1)}, Group 1 size: {len(preds_1_model_1)}")
            if len(preds_0_model_1) > 1 and len(preds_1_model_1) > 1:
                # Evaluate metrics for both models
                for model_name, preds_0, preds_1 in [
                    ("Model 1", preds_0_model_1, preds_1_model_1),
                    ("Model 2", preds_0_model_2, preds_1_model_2),
                ]:
                    
                    # Calculate accuracy, precision, recall, and positive prediction rate for each group
                    print(f"  {model_name} Metrics:")
                    accuracy_0 = accuracy_score(y_test[group_0.index], preds_0.round())
                    precision_0 = precision_score(y_test[group_0.index], preds_0.round())
                    recall_0 = recall_score(y_test[group_0.index], preds_0.round())

                    accuracy_1 = accuracy_score(y_test[group_1.index], preds_1.round())
                    precision_1 = precision_score(y_test[group_1.index], preds_1.round())
                    recall_1 = recall_score(y_test[group_1.index], preds_1.round())

                    # Calculate positive prediction rate
                    positive_rate_0 = preds_0.mean()
                    positive_rate_1 = preds_1.mean()

                    print(f"    Group 0 Accuracy: {accuracy_0:.4f}, Precision: {precision_0:.4f}, Recall: {recall_0:.4f}")
                    print(f"    Group 1 Accuracy: {accuracy_1:.4f}, Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")
                    print(f"    Positive prediction rate for Group 0: {positive_rate_0:.4f}")
                    print(f"    Positive prediction rate for Group 1: {positive_rate_1:.4f}\n")
            else:
                print("  Not enough samples in one or both groups. Skipping this combination.\n")
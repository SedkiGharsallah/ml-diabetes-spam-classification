from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

def load_and_explore_data(filepath, target_column=None, feature_names=None):
    """
    Load a CSV file and provide basic exploration statistics.
    """
    df = pd.read_csv(filepath, names=feature_names)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nNumber of samples: {df.shape[0]:,}")
    print(f"Number of features: {df.shape[1]}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  Found {missing.sum()} missing values")
        print("\nMissing values by column:")
        print(missing[missing > 0])
    else:
        print("\n No missing values found")
    
    # Target distribution if specified
    if target_column and target_column in df.columns:
        print(f"\nTarget Variable Distribution ({target_column}):")
        print(df[target_column].value_counts())
        print("\nPercentage Distribution:")
        print(df[target_column].value_counts(normalize=True) * 100)
    
    return df


def visualize_target_distribution(df):
  # Visualize target variable distribution
    fig = make_subplots(rows=1, cols=2, 
                      subplot_titles=('Count Distribution', 'Percentage Distribution'),
                      specs=[[{'type':'bar'}, {'type':'pie'}]])

  # Bar chart
    counts = df['Diabetes_binary'].value_counts()
    fig.add_trace(
      go.Bar(x=['No Diabetes', 'Diabetes'], y=counts.values, 
            marker_color=['#2ecc71', '#e74c3c'],
            text=counts.values, textposition='auto'),
      row=1, col=1
  )

    # Pie chart
    fig.add_trace(
      go.Pie(labels=['No Diabetes', 'Diabetes'], values=counts.values,
            marker_colors=['#2ecc71', '#e74c3c']),
      row=1, col=2
  )
    fig.update_layout(title_text="Diabetes Distribution", showlegend=False, height=400)
    fig.show()
def balance_data_smote(X, y, random_state=42):
    """
    Balance the dataset using SMOTE.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f" SMOTE applied: {X.shape[0]:,} → {X_resampled.shape[0]:,} samples")
    
    return X_resampled, y_resampled

def plot_confusion_matrix(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy = {acc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

def heatmap(df) :
    plt.figure(figsize=(16, 12))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix of All Features (and Target)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
  
def permutation_importance_numpy(pipe, X, y, n_repeats=20, random_state=None):
    """
    Compute permutation importance (drop in accuracy) for a fitted model `pipe`.
    - pipe: fitted estimator with method .predict(X) that accepts numpy arrays (shape (n_samples, n_features)).
    - X: numpy array, shape (n_samples, n_features).
    - y: numpy array, shape (n_samples,) (true labels).
    - n_repeats: number of random permutations per feature (average & std).
    - random_state: int or None
    Returns: Importance, defined as base_accuracy - permuted_accuracy, averaged over n_repeats runs
    The standard devation can be returned as well.
    """

    np.random.seed(random_state)
    original_score = pipe.score(X,y)
    mean=np.zeros(X.shape[1],dtype=float)
    std=np.zeros(X.shape[1],dtype=float)
    for i in range(X.shape[1]) :
      score_drop=[]
      for j in range(n_repeats):
        X_new = X.copy()
        X_new[:,i] = np.random.permutation(X_new[:,i])
        score_drop.append(original_score-pipe.score(X_new,y))
      mean[i]=(np.mean(score_drop))
      std[i]=(np.std(score_drop))


    return mean,std

def select_k_best_important_features(X, feature_names, importances, k):
    """
    Select the k most important features based on permutation importance.

    """
    indices = np.argsort(importances)[-k:][::-1]
    X_reduced = X[:, indices]
    selected_feature_names = [feature_names[i] for i in indices]
    return X_reduced, selected_feature_names
    

def knn_crossval(Xn_train, y_train,metric="accuracy"):
    K=[1,5,10,15]
    n_splits = 10
    metric_val = np.zeros((len(K),n_splits))

    cv = StratifiedKFold(n_splits=n_splits)
    for i, k in enumerate(K):
        knn = KNeighborsClassifier(n_neighbors = k)
        metric_val[i,:] = cross_val_score(knn, Xn_train, y_train, cv=cv, scoring=metric)
        print(f'K = {k}: test {metric} = {metric_val[i,:].mean():.4f} +- {metric_val[i,:].std():.4f} ({metric_val[i,:].min():.4f} - {metric_val[i,:].max():.4f})')
    return metric_val
#make the metric to choose by it an input 
def random_forest_tune_parameters(X_train, y_train, X_val, y_val, 
                       n_estimators_list=[1,3,5,10,20,50, 100], 
                       max_depth_list=[5, 10, 15, 20], 
                       min_samples_split_list=[2, 5, 10],
                       random_state=42,metric="accuracy"):
    """
    Tune Random Forest hyperparameters using a validation set.
    """
    
    results = []
    best_score = 0
    best_params = {}
    
    total_combinations = len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list)
    current = 0
    
    for n_est in n_estimators_list:
        for max_d in max_depth_list:
            for min_split in min_samples_split_list:
                current += 1
                
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                y_val_pred = rf.predict(X_val)
                val_metric = None
                if metric == 'recall':
                    val_metric = recall_score(y_val, y_val_pred)
                elif metric == 'accuracy':
                    val_metric = accuracy_score(y_val, y_val_pred)
                elif metric == 'precision' :
                    val_metric = precision_score(y_val, y_val_pred)
                else:
                    raise ValueError("Unsupported metric. Choose 'recall'  'accuracy'  'precision'.")

                results.append({
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'min_samples_split': min_split,
                    'val_metric': val_metric
                })
 
                print(f"[{current}/{total_combinations}] n_est={n_est}, max_depth={max_d}, min_split={min_split}: "
                      f"{metric}={val_metric:.4f}")
                
                if val_metric > best_score:
                    best_score = val_metric
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split
                    }
    
    results_df = pd.DataFrame(results).sort_values('val_metric', ascending=False)
    
    print(" Best Parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"   Validation {metric.capitalize()}: {best_score:.4f}")
    
    return results_df, best_params

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a machine learning model.
    """ 

    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results = {
        'model': model,
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
      }
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return results

def compare_models(results_dict, sort_by='F1-Score'):
    """
    Compare multiple models and return a sorted comparison dataframe.
    """
    comparison_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results_dict[m]['accuracy'] for m in results_dict.keys()],
        'Precision': [results_dict[m]['precision'] for m in results_dict.keys()],
        'Recall': [results_dict[m]['recall'] for m in results_dict.keys()],
        'F1-Score': [results_dict[m]['f1'] for m in results_dict.keys()],
    }).sort_values(sort_by, ascending=False)
    
    print("Model Comparison:\n")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def training(n_epochs, train_loader, valid_loader, model, criterion, optimizer, model_name): 
  device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
  train_losses, valid_losses = [], []
  valid_loss_min = np.inf

  for epoch in range(n_epochs):
      train_loss, valid_loss = 0, 0

      model.train()
      for data, label in train_loader:
          data = data.to(device=device, dtype=torch.float32)
          label = label.to(device=device, dtype=torch.long)

          optimizer.zero_grad() 
          output = model(data) 
          loss = criterion(output, label) 
          loss.backward() 
          optimizer.step() 
          train_loss += loss.item() * data.size(0) 


      model.eval()
      for data, label in valid_loader:
          data = data.to(device=device, dtype=torch.float32)
          label = label.to(device=device, dtype=torch.long)

          with torch.no_grad():
              output = model(data)
          loss = criterion(output,label)
          valid_loss += loss.item() * data.size(0)



      train_loss /= len(train_loader.sampler)
      valid_loss /= len(valid_loader.sampler)
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)

      print('epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))

      if valid_loss <= valid_loss_min:
          print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), model_name)
          valid_loss_min = valid_loss

  return train_losses, valid_losses


def evaluation(model, test_loader, criterion,nb_classes):
  device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
  test_loss = 0.0
  class_correct = list(0. for i in range(nb_classes))
  class_total = list(0. for i in range(nb_classes))

  model.eval()
  for data, label in test_loader:
      data = data.to(device=device, dtype=torch.float32)
      label = label.to(device=device, dtype=torch.long)
      with torch.no_grad():
          output = model(data)
      loss = criterion(output, label)
      test_loss += loss.item()*data.size(0)
      _, pred = torch.max(output, 1)
      correct = np.squeeze(pred.eq(label.data.view_as(pred)))
      for i in range(len(label)):
          category = label.data[i]
          class_correct[category] += correct[i].item()
          class_total[category] += 1

  test_loss = test_loss/len(test_loader.sampler)
  print('test Loss: {:.6f}\n'.format(test_loss))
  for i in range(nb_classes):
      print('test accuracy of %s: %2d%% (%2d/%2d)' % (i, 100 * class_correct[i] / (1e-4+class_total[i]), np.sum(class_correct[i]), np.sum(class_total[i])))
  print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / (1e-4+np.sum(class_total)), np.sum(class_correct), np.sum(class_total)))
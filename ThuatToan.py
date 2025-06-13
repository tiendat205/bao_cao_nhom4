import pandas as pd
import numpy as np
from collections import Counter

# Đọc dữ liệu
data = pd.read_csv(r'C:\Users\ADMIN\Downloads\Tri tue nhan tao\archive\diabetes.csv')

# Xử lý giá trị 0 ở các cột liên quan
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_impute:
    data[col] = data[col].replace(0, data[col].median())

# Gán tên đặc trưng và nhãn
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'



# -------- ID3 ALGORITHM IMPLEMENTATION --------
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())


def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[feature] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy


def best_split(data, features, target):
    gains = {f: info_gain(data, f, target) for f in features}
    return max(gains, key=gains.get)


def majority_class(y):
    return Counter(y).most_common(1)[0][0]


def build_tree(data, features, target, depth=0, max_depth=4):
    if len(set(data[target])) == 1:
        return data[target].iloc[0]
    if not features or depth == max_depth:
        return majority_class(data[target])

    best_feature = best_split(data, features, target)
    tree = {best_feature: {}}

    for val in sorted(data[best_feature].unique()):
        subset = data[data[best_feature] == val]
        if subset.empty:
            tree[best_feature][val] = majority_class(data[target])
        else:
            tree[best_feature][val] = build_tree(
                subset, [f for f in features if f != best_feature], target, depth + 1, max_depth
            )
    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample[root]
    if value in tree[root]:
        return predict(tree[root][value], sample)
    else:
        return 0


# Phân loại đặc trưng số
def discretize(df):
    df = df.copy()
    df['Glucose'] = pd.cut(df['Glucose'], bins=[0, 139, 199, float('inf')], labels=['Normal', 'Prediabetes', 'Diabetes'])
    df['BloodPressure'] = pd.cut(df['BloodPressure'], bins=[0, 79, 89, float('inf')], labels=['Normal', 'PreHT', 'HT'])
    df['BMI'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, float('inf')], labels=['Under', 'Normal', 'Over', 'Obese'])
    df['Age'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, float('inf')], labels=['20s', '30s', '40s', '50s', '60+'])
    df['SkinThickness'] = pd.cut(df['SkinThickness'], bins=3, labels=['Thin', 'Medium', 'Thick'])
    df['Insulin'] = pd.cut(df['Insulin'], bins=3, labels=['Low', 'Mid', 'High'])
    df['Pregnancies'] = pd.cut(df['Pregnancies'], bins=[-1, 0, 3, 6, float('inf')], labels=['0', '1-3', '4-6', '7+'])
    df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=3, labels=['Low', 'Med', 'High'])
    return df


data_discrete = discretize(data)

# Chia tập train-test
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data_discrete, test_size=0.2, random_state=42)

# Huấn luyện cây
tree_id3 = build_tree(train_data, features, target)

# Đánh giá trên tập test
correct = 0
for _, row in test_data.iterrows():
    pred = predict(tree_id3, row)
    if pred == row[target]:
        correct += 1

accuracy = correct / len(test_data)
print(f"\nĐộ chính xác trên tập test: {accuracy:.2f}")

# -------- DỰ ĐOÁN MINH HỌA --------
print("\n--- DỰ ĐOÁN CHO BỆNH NHÂN MỚI ---")
new_patient = {
    'Pregnancies': 2,
    'Glucose': 160,
    'BloodPressure': 78,
    'SkinThickness': 22,
    'Insulin': 85,
    'BMI': 27.5,
    'DiabetesPedigreeFunction': 0.6,
    'Age': 35
}

new_data = discretize(pd.DataFrame([new_patient]))
result = predict(tree_id3, new_data.iloc[0])
diagnosis = " Bị tiểu đường" if result == 1 else " Không bị tiểu đường"
print("Thông tin bệnh nhân:")
new_data = pd.DataFrame([new_patient])[features]
print(new_data.to_string(index=False))
print(f"=> Kết luận: {diagnosis}")

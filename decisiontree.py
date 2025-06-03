import matplotlib
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree


matplotlib.use('TkAgg')  # 또는 'QtAgg', 'Agg'는 비추천 (화면 없음)

# 1. 데이터 로딩
CSV_PATH = '../../dataset/masan_100/masan_100_ade20k_summary.csv'
df = pd.read_csv(CSV_PATH)

# 2. 분류 or 회귀 선택
MODE = 'classification'  # 또는 'regression'
TARGET = 'LP_lev' if MODE == 'classification' else 'LP_sc'

# 3. 전처리
X = df.drop(columns=['segment_ID', 'LP_lev', 'LP_sc'])  # 독립변수
y = df[TARGET]  # 종속변수

# (선택) 분류용: y가 문자열일 경우 숫자로 변환
if MODE == 'classification' and y.dtype == object:
    y = y.astype('category').cat.codes

# 4. 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 학습
if MODE == 'classification':
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
else:
    model = DecisionTreeRegressor(max_depth=4, random_state=42)

model.fit(X_train, y_train)

# 6. 예측 및 평가
y_pred = model.predict(X_test)

if MODE == 'classification':
    print(classification_report(y_test, y_pred))
else:
    print("MSE:", mean_squared_error(y_test, y_pred))

# 7. 결정 트리 시각화 (텍스트 출력)
print(export_text(model, feature_names=list(X.columns)))

# 8. 시각적 트리 플롯 저장
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.savefig("decision_tree_plot.png")
plt.show()
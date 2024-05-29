import pandas as pd
from catboost import CatBoostClassifier

output_path = "./output/"
def predict(test: pd.DataFrame) -> None:
    """Делает предикт и сохраняет результат в csv-файл"""
    clf_boost = CatBoostClassifier()
    clf_boost.load_model("./model/my_cat_boost.cbm")
    predicts = clf_boost.predict(test.drop(columns=['client_id']))
    
    submission = pd.DataFrame({
    'client_id':test["client_id"],
    'preds': predicts
    })
    submission.set_index("client_id").to_csv(output_path + "submission.csv")
    
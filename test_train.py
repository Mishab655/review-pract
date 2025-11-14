from train import train
import joblib 

def test_train():
    train()
    model = joblib.load('model.pkl')
    assert model is not None
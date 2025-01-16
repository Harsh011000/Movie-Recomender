import os
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import keras
from tensorflow.keras.layers import Layer

@keras.saving.register_keras_serializable(package="MyLayers")
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

custom_objects = {
    "L2Normalization": L2Normalization
}

model = tf.keras.models.load_model('Models/my_model.keras',custom_objects=custom_objects)

user_scaler=load('Models/UserScaler.joblib')
movie_scaler=load('Models/MovieScaler.joblib')
rating_scaler=load('Models/MinMaxRatingScaler.joblib')

def load_dataset(scalerItem,scalerUser):
    # Load the CSV file
    df = pd.read_csv("Api/updated_api_data.csv")
    dfUser=pd.read_csv("Api/user_file_forApi.csv")
    item_train = scalerItem.transform(df.iloc[:,3:20])
    user_train= scalerUser.transform(dfUser)
    info=df.iloc[:,[1,2,4,20]]
    return [info,user_train[:,3:],item_train[:,2:]]

from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf  # Or the library you're using for your model

app = Flask(__name__)

# Load your pre-trained model
#model = tf.keras.models.load_model("path_to_your_model.h5")  # Update the path to your model file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON request
        input_data = request.json.get("data")
        
        # Ensure input data is a valid numpy array
        if input_data is None or not isinstance(input_data, list):
            return jsonify({"error": "Invalid input. Expected a list of numbers."}), 400
        
        # Convert input data to a NumPy array
        input_array = np.array(input_data).reshape(1, -1)  # Adjust reshape based on model input requirements

        columns = ['3.95', '4.25', '0.0', '0.0.1', '4.0.1',
           '4.12', '4.0.2', '4.04', '0.0.2', '3.0', '4.0.3', '0.0.3', '3.88', '3.89']
        df = pd.DataFrame(input_array, columns=columns)
        tempDf=pd.read_csv('Api/temp_user_data.csv')
        df=pd.concat([tempDf,df],axis=1)
        
        # Replicate rows until the length is 847
        replicated_df = pd.concat([df] * (847 // len(df)), ignore_index=True)  # Full replication
        remaining_rows = 847 - len(replicated_df)  # Remaining rows to reach 847
        if remaining_rows > 0:
            replicated_df = pd.concat([replicated_df, df.iloc[:remaining_rows]], ignore_index=True)

        user_train=user_scaler.transform(replicated_df)


        info,_,item_train=load_dataset(movie_scaler,user_scaler)
        
        # Make predictions using the model
        predictions = model.predict([user_train,item_train]) #[:,3:]
        predictions= rating_scaler.inverse_transform(predictions)
        
        # Convert predictions to a list
        predictions_list = predictions.flatten().tolist()  # Flatten if predictions are multi-dimensional
        
        # Format predictions with additional dataframe information
        formatted_predictions = []
        for i, pred in enumerate(predictions_list):
            if i >= len(info):  # Ensure we don't access out-of-range indices
                break
            
            # Get the corresponding row from the dataframe
            row = info.iloc[i]
            label = ""
            if pred >= 4.5:
                label = "Highly Recommended"
            elif 3.5 <= pred < 4.5:  # Combine conditions correctly
                label = "Good Movie for You"
            elif 2.5 <= pred < 3.5:
                label = "Average"
            else:
                label = "Not for You"          
            # Build the JSON object for the prediction
            prediction_object = {
                "prediction": pred,
                "label":label,
                "title": row["title"],
                "genres": row["genres"],
                "year": int(row["2003"]),  # Adjust the column name if needed
                "image_url": row["Image URL"]
            }
            formatted_predictions.append(prediction_object)
        
        sorted_data = sorted(formatted_predictions, key=lambda x: x["prediction"], reverse=True)
        
        # Return sorted predictions as a JSON object
        return jsonify(sorted_data)
        
        # Return formatted predictions as a JSON object
        #return jsonify(formatted_predictions)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


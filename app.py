from flask import Flask, request, render_template
import pickle
import pandas as pd
import qwerty

app = Flask(__name__)

main_df = pd.read_excel("data.xlsx")

# Load the model
model = pickle.load(open('model_pred.pkl', 'rb'))


@app.route('/', methods=["GET", "POST"])
def gfg():
    if request.method == "POST":
        cuisines = request.form.get("cuisines")
        current_location = request.form.get("currentlocation")
        location = request.form.get("location")
        dishcategory = request.form.get("dishcategory")
        res_name = qwerty.abc(cuisines, location, current_location, dishcategory)

        # Create a list of dictionaries with the necessary data
        res_data = []
        for res in res_name:
            row = main_df.loc[main_df['res_name'] == res]
            if not row.empty:
                res_data.append({
                    'res_name': res,
                    'rating': row['rating'].values[0],
                    'cusines': row['cusines'].values[0],
                    'res_link': row['res_link'].values[0]
                })

        return render_template("index.html", res_data=res_data, cuisines=cuisines)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
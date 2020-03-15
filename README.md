# README File

**Step 1:**
Run `pip install -r requirements.txt` to install all the dependencies to run the source python file

**Step 2:**
Navigate to the **src** folder and run the `Niki.ai_Assignment_Part1.py` file

- An 80:20 train:test split cross-validation was done to train the model
- Used a RandomForestClassifier to classify the different classes, which in this case are the `Star Ratings`. By using this algorithm, one need not normalize the data. 
- Used Tf-Idf Vectorizer to build the feature vector consisting of important words 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Step 1: Data Preparation

# Load dataset
data = pd.read_csv('telcoChurn.csv')

# Data preprocessing
# Drop the target column 'Churn' from features and save it as the label
X = data.drop(['Churn'], axis=1)
y = data['Churn']

# Convert string labels to numerical values ('Yes'/'No' -> 1/0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # This converts 'Yes' to 1 and 'No' to 0

# Identify and drop non-numeric columns (like customer ID)
X = X.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train)  # Convert 1/0 to one-hot encoding [1, 0] and [0, 1]
y_test = to_categorical(y_test)

# Step 2: Build and Compile the Model
model = Sequential()

# Input layer with 64 neurons
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))  # Add dropout layer after the input layer

# Hidden layer with 64 neurons
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))  # Add dropout after hidden layer

# Hidden layer with 32 neurons
model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))

# Output layer for binary classification
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 3: Train the Model with ReduceLROnPlateau and EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[reduce_lr, early_stopping])

# Step 4: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

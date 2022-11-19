{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f2bdc75e",
   "metadata": {},
   "source": [
    "# Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9f56babd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f4eb5cd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from lightgbm import LGBMRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd2a2b44",
   "metadata": {},
   "source": [
    "# Import Preprocessed Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8229eca4",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"C:/Users/Rajesh/Downloads/autos_preprocessed.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d334ae1d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>price</th>\n",
       "      <th>vehicleType</th>\n",
       "      <th>yearOfRegistration</th>\n",
       "      <th>gearbox</th>\n",
       "      <th>powerPS</th>\n",
       "      <th>model</th>\n",
       "      <th>kilometer</th>\n",
       "      <th>monthOfRegistration</th>\n",
       "      <th>fuelType</th>\n",
       "      <th>brand</th>\n",
       "      <th>notRepairedDamage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>18300</td>\n",
       "      <td>coupe</td>\n",
       "      <td>2011</td>\n",
       "      <td>manual</td>\n",
       "      <td>190</td>\n",
       "      <td>not-declared</td>\n",
       "      <td>125000</td>\n",
       "      <td>5</td>\n",
       "      <td>diesel</td>\n",
       "      <td>audi</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>9800</td>\n",
       "      <td>suv</td>\n",
       "      <td>2004</td>\n",
       "      <td>automatic</td>\n",
       "      <td>163</td>\n",
       "      <td>grand</td>\n",
       "      <td>125000</td>\n",
       "      <td>8</td>\n",
       "      <td>diesel</td>\n",
       "      <td>jeep</td>\n",
       "      <td>not-declared</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1500</td>\n",
       "      <td>small car</td>\n",
       "      <td>2001</td>\n",
       "      <td>manual</td>\n",
       "      <td>75</td>\n",
       "      <td>golf</td>\n",
       "      <td>150000</td>\n",
       "      <td>6</td>\n",
       "      <td>petrol</td>\n",
       "      <td>volkswagen</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>3600</td>\n",
       "      <td>small car</td>\n",
       "      <td>2008</td>\n",
       "      <td>manual</td>\n",
       "      <td>69</td>\n",
       "      <td>fabia</td>\n",
       "      <td>90000</td>\n",
       "      <td>7</td>\n",
       "      <td>diesel</td>\n",
       "      <td>skoda</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>650</td>\n",
       "      <td>limousine</td>\n",
       "      <td>1995</td>\n",
       "      <td>manual</td>\n",
       "      <td>102</td>\n",
       "      <td>3er</td>\n",
       "      <td>150000</td>\n",
       "      <td>10</td>\n",
       "      <td>petrol</td>\n",
       "      <td>bmw</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  price vehicleType  yearOfRegistration    gearbox  powerPS  \\\n",
       "0           1  18300       coupe                2011     manual      190   \n",
       "1           2   9800         suv                2004  automatic      163   \n",
       "2           3   1500   small car                2001     manual       75   \n",
       "3           4   3600   small car                2008     manual       69   \n",
       "4           5    650   limousine                1995     manual      102   \n",
       "\n",
       "          model  kilometer  monthOfRegistration fuelType       brand  \\\n",
       "0  not-declared     125000                    5   diesel        audi   \n",
       "1         grand     125000                    8   diesel        jeep   \n",
       "2          golf     150000                    6   petrol  volkswagen   \n",
       "3         fabia      90000                    7   diesel       skoda   \n",
       "4           3er     150000                   10   petrol         bmw   \n",
       "\n",
       "  notRepairedDamage  \n",
       "0               Yes  \n",
       "1      not-declared  \n",
       "2                No  \n",
       "3                No  \n",
       "4               Yes  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2602d158",
   "metadata": {},
   "source": [
    "# Label Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "68d4e1ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['price', 'yearOfRegistration', 'powerPS', 'kilometer',\n",
      "       'monthOfRegistration', 'gearbox_labels', 'notRepairedDamage_labels',\n",
      "       'model_labels', 'brand_labels', 'fuelType_labels',\n",
      "       'vehicleType_labels'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']\n",
    "\n",
    "mapper = {}\n",
    "for i in labels:\n",
    "    mapper[i] = LabelEncoder()\n",
    "    mapper[i].fit(data[i])\n",
    "    tr = mapper[i].transform(data[i])\n",
    "    np.save(str('classes'+i+'.npy'), mapper[i].classes_)\n",
    "    data.loc[:, i+'_labels'] = pd.Series(tr, index=data.index)\n",
    "    \n",
    "labeled = data[['price', 'yearOfRegistration','powerPS','kilometer','monthOfRegistration']\n",
    "                  +[x+\"_labels\" for x in labels]]\n",
    "\n",
    "print(labeled.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0af9ede",
   "metadata": {},
   "source": [
    "# Different Metrics Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2c9861d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_scores(Y_actual, Y_pred, X_train):\n",
    "    scores = dict()\n",
    "    mae = mean_absolute_error(Y_actual, Y_pred)\n",
    "    mse = mean_squared_error(Y_actual, Y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    rmsle = np.log(rmse)\n",
    "    r2 = r2_score(Y_actual, Y_pred)\n",
    "    n, k = X_train.shape\n",
    "    adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))\n",
    "    \n",
    "    scores['mae']=mae\n",
    "    scores['mse']=mse\n",
    "    scores['rmse']=rmse\n",
    "    scores['rmsle']=rmsle\n",
    "    scores['r2']=r2\n",
    "    scores['adj_r2_score']=adj_r2_score\n",
    "    \n",
    "    return scores"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b101b979",
   "metadata": {},
   "source": [
    "# Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "da8126ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = labeled.iloc[:,1:].values\n",
    "Y = labeled.iloc[:,0].values.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0f6f18db",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ad8c48e",
   "metadata": {},
   "source": [
    "# Predictive Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1991ead",
   "metadata": {},
   "source": [
    "# LGBM Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "bed8905e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Rajesh\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'mae': 1327.549477341283,\n",
       " 'mse': 9492244.283543464,\n",
       " 'rmse': 3080.948601249859,\n",
       " 'rmsle': 8.032992815968017,\n",
       " 'r2': 0.8668348937732229,\n",
       " 'adj_r2_score': 0.8668269262555739}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LGBMRegressor(boosting_type=\"gbdt\",learning_rate=0.07,metric=\"rmse\",n_estimators=300,objective=\"root_mean_squared_error\",random_state=42,reg_sqrt=True)\n",
    "\n",
    "model.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred = model.predict(X_test)\n",
    "\n",
    "find_scores(Y_test, Y_pred, X_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f7911de",
   "metadata": {},
   "source": [
    "# Save Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ed4e4c11",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(model, open('resale_model.sav', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44111ba4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}